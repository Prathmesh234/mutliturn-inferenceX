import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_act_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                       M, K: tl.constexpr, N: tl.constexpr,
                       ACT: tl.constexpr, NEG_SLOPE,
                       BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N)
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k in range(K):
        x_k = tl.load(x_ptr + offs_m * K + k, mask=mask_m, other=0.0)
        w_k = tl.load(w_ptr + offs_n * K + k)
        acc += x_k[:, None] * w_k[None, :]
    b = tl.load(b_ptr + offs_n)
    acc += b[None, :]
    if ACT == 1:
        acc = tl.where(acc > 0, acc, 0.0)
    elif ACT == 2:
        acc = tl.where(acc > 0, acc, acc * NEG_SLOPE)
    elif ACT == 3:
        acc = (tl.exp(2 * acc) - 1) / (tl.exp(2 * acc) + 1)
    out_off = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_m[:, None])


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class LinearBlockNew(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlockNew, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        self._act_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)

        self._fuse = self.norm is None and self._act_name in ('relu', 'lrelu', 'tanh', 'none')
        self._act_code = {'relu': 1, 'lrelu': 2, 'tanh': 3, 'none': 0}.get(self._act_name, 0) if self._fuse else 0
        self._neg_slope = 0.2 if self._act_name == 'lrelu' else 0.0
        self.K = input_dim
        self.N = output_dim

    def forward(self, x):
        K = self.K
        N = self.N
        x_flat = x.reshape(-1, K)
        M = x_flat.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64),)
        _linear_act_kernel[grid](x_flat, self.fc.weight, self.fc.bias, out,
                                 M, K, N, self._act_code, self._neg_slope,
                                 BLOCK_M=64, num_warps=1)
        out = out.reshape(*x.shape[:-1], N)
        if not self._fuse:
            if self.norm:
                out = self.norm(out)
            if self.activation:
                out = self.activation(out)
        return out
