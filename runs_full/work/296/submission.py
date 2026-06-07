import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _simse_kernel(pred_ptr, real_ptr, out_ptr, n_elements, inv_n2, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for start in range(0, n_elements, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_elements
        p = tl.load(pred_ptr + idx, mask=mask, other=0.0)
        r = tl.load(real_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(r - p, axis=0)
    res = acc * acc * inv_n2
    tl.store(out_ptr, res)


class SIMSENew(nn.Module):
    def __init__(self):
        super(SIMSENew, self).__init__()

    def forward(self, pred, real):
        pred = pred.contiguous()
        real = real.contiguous()
        n = pred.numel()
        out = torch.empty((), device=pred.device, dtype=torch.float32)
        BLOCK_SIZE = 256
        _simse_kernel[(1,)](pred, real, out, n, 1.0 / (n ** 2), BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
