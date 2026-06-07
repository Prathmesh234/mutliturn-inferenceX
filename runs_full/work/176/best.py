import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _nae_kernel(x_ptr, fc_ptr, bf_ptr, out_ptr, N, F,
                stride_xn, stride_xf,
                BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    xptrs = x_ptr + row * stride_xn + offs * stride_xf
    x = tl.load(xptrs, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)
    dot = tl.sum(x * fc, axis=0) + tl.load(bf_ptr)
    h = 1.0 / (1.0 + tl.exp(-dot))
    pos = tl.where(x < 0, 0.0, x)
    neg = tl.where(x > 0, 0.0, x)
    out = pos + h * neg
    tl.store(out_ptr + row * stride_xn + offs * stride_xf, out, mask=mask)


class NodeAdaptiveEncoderNew(nn.Module):
    def __init__(self, num_features, dropout=0.5):
        super().__init__()
        self.fc = nn.Parameter(torch.zeros(size=(num_features, 1)))
        nn.init.xavier_normal_(self.fc.data, gain=1.414)
        self.bf = nn.Parameter(torch.zeros(size=(1,)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        N, F = x.shape
        out = torch.empty_like(x)
        BLOCK_F = triton.next_power_of_2(F)
        grid = (N,)
        _nae_kernel[grid](x, self.fc, self.bf, out, N, F,
                          x.stride(0), x.stride(1),
                          BLOCK_F=BLOCK_F, num_warps=4)
        return out
