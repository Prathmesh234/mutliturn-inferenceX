import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l2_kernel(out_ptr, tgt_ptr, res_ptr, T, C, R, inv_T,
               BLOCK: tl.constexpr):
    t = tl.arange(0, BLOCK)
    mask = t < T
    n = t // R
    r = t % R
    base = n * (C * R) + r
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(C):
        offs = base + c * R
        o = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(tgt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d = o - g
        acc += d * d
    norm = tl.sqrt(acc)
    norm = tl.where(mask, norm, 0.0)
    res = tl.sum(norm) * inv_T
    tl.store(res_ptr, res)


class L2New(nn.Module):
    def __init__(self):
        super(L2New, self).__init__()

    def forward(self, output, target):
        output = output.contiguous()
        target = target.contiguous()
        d0 = output.shape[0]
        C = output.shape[1]
        R = 1
        for s in output.shape[2:]:
            R *= s
        T = d0 * R
        res = torch.empty((), device=output.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(T)
        _l2_kernel[(1,)](output, target, res, T, C, R, 1.0 / T,
                         BLOCK=BLOCK, num_warps=1)
        return res.to(output.dtype)


