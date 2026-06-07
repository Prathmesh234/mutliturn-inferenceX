import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.
            weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])


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


@triton.jit
def _conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW, stride,
    K,
    ACT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per output spatial position (n, oh, ow)
    m = tl.program_id(0)
    ow = m % OW
    tmp = m // OW
    oh = tmp % OH
    n = tmp // OH

    oc = tl.arange(0, BLOCK_OC)
    oc_mask = oc < OC

    k = tl.arange(0, BLOCK_K)
    k_mask = k < K
    ic = k // (KH * KW)
    krem = k % (KH * KW)
    kh = krem // KW
    kw = krem % KW
    ih = oh * stride + kh
    iw = ow * stride + kw
    x_base = n * IC * IH * IW
    x_off = x_base + ic * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)  # [BLOCK_K]

    w_off = oc[:, None] * K + k[None, :]
    w_val = tl.load(w_ptr + w_off, mask=oc_mask[:, None] & k_mask[None, :],
                    other=0.0)  # [BLOCK_OC, BLOCK_K]

    acc = tl.sum(w_val * x_val[None, :], axis=1)  # [BLOCK_OC]

    bias = tl.load(b_ptr + oc, mask=oc_mask, other=0.0)
    acc += bias

    if ACT == 1:  # relu
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:  # lrelu 0.2
        acc = tl.where(acc > 0, acc, acc * 0.2)
    elif ACT == 3:  # tanh
        acc = (tl.exp(2 * acc) - 1) / (tl.exp(2 * acc) + 1)

    out_base = n * OC * OH * OW + oh * OW + ow
    out_off = out_base + oc * OH * OW
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


class Conv2dBlockNew(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
        norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlockNew, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
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
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
            bias=self.use_bias)

        # determine fused activation code (only when no norm in between)
        self._act_code = 0
        if self.norm is None:
            if activation == 'relu':
                self._act_code = 1
            elif activation == 'lrelu':
                self._act_code = 2
            elif activation == 'tanh':
                self._act_code = 3

    def _conv_triton(self, x, act_code):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        stride = self.conv.stride[0]
        OH = (IH - KH) // stride + 1
        OW = (IW - KW) // stride + 1
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        K = IC * KH * KW
        BLOCK_OC = triton.next_power_of_2(OC)
        BLOCK_K = triton.next_power_of_2(K)
        grid = (N * OH * OW,)
        _conv2d_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, stride,
            K,
            act_code,
            BLOCK_OC=BLOCK_OC,
            BLOCK_K=BLOCK_K,
            num_warps=1,
        )
        return out

    def forward(self, x):
        x = self.pad(x)
        if self.norm is None:
            # fuse conv + bias + activation in one kernel
            return self._conv_triton(x, self._act_code)
        else:
            x = self._conv_triton(x, 0)
            x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            return x
