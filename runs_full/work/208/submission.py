import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dice_all(in_ptr, tgt_ptr, out_ptr, B, N, HW,
              BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_P)
    mb = offs_b < B
    mn = offs_n < N
    mp = offs_p < HW
    mask = mb[:, None, None] & mn[None, :, None] & mp[None, None, :]
    base = (offs_b[:, None, None] * (N * HW)
            + offs_n[None, :, None] * HW
            + offs_p[None, None, :])
    x = tl.load(in_ptr + base, mask=mask, other=-float('inf'))
    t = tl.load(tgt_ptr + base, mask=mask, other=0.0)
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None, :])
    s = tl.sum(e, axis=1)
    soft = e / s[:, None, :]
    soft = tl.where(mask, soft, 0.0)
    inter = tl.sum(tl.sum(soft * t, axis=2), axis=1)
    card = tl.sum(tl.sum(soft + t, axis=2), axis=1)
    eps = 1e-06
    dice = (2.0 * inter + 1.0) / (card + 1.0 + eps)
    dice = tl.where(mb, dice, 0.0)
    res = tl.sum(dice) / B
    tl.store(out_ptr, res)


class DiceLoss(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(DiceLoss, self).__init__()
        self.eps: 'float' = 1e-06
        self.dims = dims


class DiceNew(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(DiceNew, self).__init__()
        self.dice_loss = DiceLoss(dims)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
                weights=None) -> torch.Tensor:
        input = input.contiguous()
        target = target.contiguous()
        B, N, H, W = input.shape
        HW = H * W
        out = torch.empty((), device=input.device, dtype=torch.float32)
        _dice_all[(1,)](input, target, out, B, N, HW,
                        BLOCK_B=triton.next_power_of_2(B),
                        BLOCK_N=triton.next_power_of_2(N),
                        BLOCK_P=triton.next_power_of_2(HW), num_warps=1)
        return out.to(input.dtype)
