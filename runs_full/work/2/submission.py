import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, scale_ptr, bias_ptr, out_ptr,
                 N1, inner_size, N3,
                 BLOCK_N1: tl.constexpr):
    pid = tl.program_id(0)
    n0 = pid // inner_size
    inner = pid % inner_size
    n3 = inner % N3
    n1 = tl.arange(0, BLOCK_N1)
    mask = n1 < N1
    offs = n0 * N1 * inner_size + n1 * inner_size + inner
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(sumsq)
    s = tl.load(scale_ptr + n3).to(tl.float32)
    b = tl.load(bias_ptr + n3).to(tl.float32)
    out = x / norm * s + b
    tl.store(out_ptr + offs, out, mask=mask)


class CustomizeLayerNew(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))

    def forward(self, x):
        N0 = x.shape[0]
        N1 = x.shape[1]
        inner_size = x[0, 0].numel()
        N3 = x.shape[-1]
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK_N1 = triton.next_power_of_2(N1)
        grid = (N0 * inner_size,)
        _norm_kernel[grid](xc, self.scale, self.bias, out,
                           N1, inner_size, N3, BLOCK_N1=BLOCK_N1, num_warps=1)
        return out

    def __repr__(self):
        return 'CustomizedLayer(in_dim=%d)' % self.in_dim
