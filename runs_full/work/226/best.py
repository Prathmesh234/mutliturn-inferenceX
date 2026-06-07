import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _all_kernel(in_ptr, tgt_ptr, out_ptr, B, N, HW, numel, eps,
                BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HW: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hw = tl.arange(0, BLOCK_HW)
    mb = offs_b < B
    mn = offs_n < N
    mhw = offs_hw < HW
    valid = (mb[:, None, None] & mn[None, :, None]) & mhw[None, None, :]
    ptrs = offs_b[:, None, None] * (N * HW) + offs_n[None, :, None] * HW + offs_hw[None, None, :]
    x = tl.load(in_ptr + ptrs, mask=valid, other=-float('inf'))
    t = tl.load(tgt_ptr + ptrs, mask=valid, other=0.0)
    m = tl.max(x, axis=1)[:, None, :]
    e = tl.exp(x - m)
    s = tl.sum(e, axis=1)[:, None, :]
    soft = e / s
    soft = tl.where(valid, soft, 0.0)
    inter = tl.sum(tl.sum(soft * t, axis=2), axis=1)
    card = tl.sum(tl.sum(soft + t, axis=2), axis=1)
    diff = soft - t
    sq2 = tl.where(valid, diff * diff, 0.0)
    dice_b = 1.0 - (2.0 * inter + 1.0) / (card + 1.0 + eps)
    dice_b = tl.where(mb, dice_b, 0.0)
    dice = tl.sum(dice_b, axis=0) / B
    mse = tl.sum(tl.sum(tl.sum(sq2, axis=2), axis=1), axis=0) / numel
    res = mse * 10.0 + dice
    tl.store(out_ptr, res)


class MSEDICELossNew(nn.Module):
    def __init__(self) -> None:
        super(MSEDICELossNew, self).__init__()
        self.eps = 1e-06

    def forward(self, input: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        input = input.contiguous()
        target = target.contiguous().to(input.dtype)
        B, N, H, W = input.shape
        HW = H * W
        numel = B * N * HW
        out = torch.empty((), device=input.device, dtype=torch.float32)
        _all_kernel[(1,)](
            input, target, out, B, N, HW, numel, self.eps,
            BLOCK_B=triton.next_power_of_2(B),
            BLOCK_N=triton.next_power_of_2(N),
            BLOCK_HW=triton.next_power_of_2(HW),
            num_warps=1,
        )
        return out
