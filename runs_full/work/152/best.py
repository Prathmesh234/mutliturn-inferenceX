import torch
import triton
import triton.language as tl


@triton.jit
def _gmp_kernel(x_ptr, out_ptr, R, BLOCK_R: tl.constexpr):
    pid = tl.program_id(axis=0)
    base = pid * R
    offs = tl.arange(0, BLOCK_R)
    mask = offs < R
    x = tl.load(x_ptr + base + offs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    tl.store(out_ptr + pid, m)


class GlobalMaxPoolNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        H = x.shape[-2]
        W = x.shape[-1]
        R = H * W
        lead = x.shape[:-2]
        n_rows = 1
        for s in lead:
            n_rows *= s
        xc = x.contiguous()
        out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        BLOCK_R = triton.next_power_of_2(R)
        _gmp_kernel[(n_rows,)](xc, out, R, BLOCK_R=BLOCK_R, num_warps=1)
        return out.view(*lead, 1, 1)
