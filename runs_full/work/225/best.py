import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _avgmax_kernel(x_ptr, out_ptr, n_rows, R, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    base = pid * R
    acc_sum = 0.0
    acc_max = -float('inf')
    for off in range(0, R, BLOCK_R):
        idx = off + tl.arange(0, BLOCK_R)
        mask = idx < R
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc_sum += tl.sum(tl.where(mask, v, 0.0), axis=0)
        acc_max = tl.maximum(acc_max, tl.max(tl.where(mask, v, -float('inf')), axis=0))
    res = acc_sum / R + acc_max
    tl.store(out_ptr + pid, res)


class AdaptiveAvgMaxPoolNew(nn.Module):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        N, C, H, W = x.shape
        R = H * W
        n_rows = N * C
        xf = x.contiguous().view(n_rows, R)
        out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        BLOCK_R = triton.next_power_of_2(R) if R < 1024 else 1024
        grid = (n_rows,)
        _avgmax_kernel[grid](xf, out, n_rows, R, BLOCK_R=BLOCK_R, num_warps=4)
        return out.view(N, C, 1, 1)
