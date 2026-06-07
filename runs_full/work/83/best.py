import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _lin_sig_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, NC: tl.constexpr,
                    BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    k = tl.arange(0, NC)
    x = tl.load(x_ptr + rows[:, None] * NC + k[None, :],
                mask=rmask[:, None], other=0.0)
    w = tl.load(w_ptr + k)
    acc = tl.sum(x * w[None, :], axis=1)
    b = tl.load(b_ptr)
    acc = acc + b
    out = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + rows, out, mask=rmask)


class BcNew(nn.Module):
    def __init__(self, nc):
        super(BcNew, self).__init__()
        self.nn = nn.Linear(nc, 1)

    def forward(self, input):
        nc = self.nn.in_features
        x = input.contiguous()
        M = x.numel() // nc
        out = torch.empty(M, device=x.device, dtype=x.dtype)
        BLOCK_M = triton.next_power_of_2(M)
        grid = (1,)
        _lin_sig_kernel[grid](x, self.nn.weight, self.nn.bias, out, M,
                              NC=nc, BLOCK_M=BLOCK_M, num_warps=2)
        return out.view(*input.shape[:-1], 1)
