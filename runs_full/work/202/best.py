import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _dice_kernel(inp_ptr, tgt_ptr, out_ptr, N, HW, eps, B,
                 BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HW: tl.constexpr):
    b = tl.arange(0, BLOCK_B)
    n = tl.arange(0, BLOCK_N)
    hw = tl.arange(0, BLOCK_HW)
    offs = b[:, None, None] * N * HW + n[None, :, None] * HW + hw[None, None, :]
    mask = (b[:, None, None] < B) & (n[None, :, None] < N) & (hw[None, None, :] < HW)
    x = tl.load(inp_ptr + offs, mask=mask, other=-float('inf'))
    t = tl.load(tgt_ptr + offs, mask=mask, other=0.0)
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None, :])
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=1)
    soft = e / denom[:, None, :]
    soft = tl.where(mask, soft, 0.0)
    tm = tl.where(mask, t, 0.0)
    inter = tl.sum(tl.sum(soft * tm, axis=2), axis=1)
    card = tl.sum(tl.sum(soft + tm, axis=2), axis=1)
    dice = (2.0 * inter + 1.0) / (card + 1.0 + eps)
    bmask = b < B
    loss = tl.sum(tl.where(bmask, 1.0 - dice, 0.0)) / B
    tl.store(out_ptr, loss)


class DiceLossNew(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super().__init__()
        self.eps: float = 1e-06
        self.dims = dims

    def forward(self, input, target, weights=None):
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxNxHxW. Got: {}'.format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError('input and target shapes must be the same. Got: {}'.format(input.shape))
        if not input.device == target.device:
            raise ValueError('input and target must be in the same device. Got: {}'.format(input.device))
        B, Nc, H, W = input.shape
        HW = H * W
        input = input.contiguous()
        target = target.contiguous()
        out = torch.empty((), device=input.device, dtype=torch.float32)
        _dice_kernel[(1,)](input, target, out, Nc, HW, self.eps, B,
                           BLOCK_B=triton.next_power_of_2(B),
                           BLOCK_N=triton.next_power_of_2(Nc),
                           BLOCK_HW=triton.next_power_of_2(HW), num_warps=1)
        return out
