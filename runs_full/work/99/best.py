import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l1_kernel(out_ptr, tgt_ptr, res_ptr, n_elements, inv_n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    o = tl.load(out_ptr + offs, mask=mask, other=0.0)
    t = tl.load(tgt_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(tl.abs(o - t)) * inv_n
    tl.store(res_ptr, s)


class L1New(nn.Module):
    def __init__(self):
        super(L1New, self).__init__()

    def forward(self, output, target):
        n = output.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        res = torch.empty((), device=output.device, dtype=torch.float32)
        _l1_kernel[(1,)](output, target, res, n, 1.0 / n, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return res
