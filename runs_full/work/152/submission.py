import torch
import triton
import triton.language as tl


@triton.jit
def _gmp_kernel(x_ptr, out_ptr, n_rows, R, ROWS: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(axis=0)
    rm = pid * ROWS + tl.arange(0, ROWS)
    rr = tl.arange(0, BLOCK_R)
    ptrs = x_ptr + rm[:, None] * R + rr[None, :]
    mask = (rm[:, None] < n_rows) & (rr[None, :] < R)
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    tl.store(out_ptr + rm, m, mask=rm < n_rows)


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
        ROWS = 4
        BLOCK_R = triton.next_power_of_2(R)
        grid = (triton.cdiv(n_rows, ROWS),)
        _gmp_kernel[grid](xc, out, n_rows, R, ROWS=ROWS, BLOCK_R=BLOCK_R, num_warps=1)
        return out.view(*lead, 1, 1)
