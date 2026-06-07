import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dice_per_batch(in_ptr, tgt_ptr, out_ptr, N, HW,
                    BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
    b = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_P)
    mask = (offs_n[:, None] < N) & (offs_p[None, :] < HW)
    base = b * N * HW + offs_n[:, None] * HW + offs_p[None, :]
    x = tl.load(in_ptr + base, mask=mask, other=-float('inf'))
    t = tl.load(tgt_ptr + base, mask=mask, other=0.0)
    # softmax over channel axis (axis 0)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m[None, :])
    s = tl.sum(e, axis=0)
    soft = e / s[None, :]
    soft = tl.where(mask, soft, 0.0)
    inter = tl.sum(tl.sum(soft * t, axis=0), axis=0)
    card = tl.sum(tl.sum(soft + t, axis=0), axis=0)
    eps = 1e-06
    dice = (2.0 * inter + 1.0) / (card + 1.0 + eps)
    tl.store(out_ptr + b, dice)


@triton.jit
def _mean(in_ptr, out_ptr, B, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < B
    v = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(v) / B)


class DiceLoss(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(DiceLoss, self).__init__()
        self.eps: 'float' = 1e-06
        self.dims = dims


class Dice(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(Dice, self).__init__()
        self.dice_loss = DiceLoss(dims)


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
        per = torch.empty(B, device=input.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_P = triton.next_power_of_2(HW)
        _dice_per_batch[(B,)](input, target, per, N, HW,
                              BLOCK_N=BLOCK_N, BLOCK_P=BLOCK_P, num_warps=4)
        out = torch.empty((), device=input.device, dtype=torch.float32)
        _mean[(1,)](per, out, B, BLOCK=triton.next_power_of_2(B))
        return out.to(input.dtype)
