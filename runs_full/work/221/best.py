import torch
import numpy as np
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _dice_fused_kernel(x_ptr, y_ptr, out_ptr, NC, C, S, smooth, DO_BG: tl.constexpr,
                       BLOCK_NC: tl.constexpr, BLOCK_S: tl.constexpr):
    rows = tl.arange(0, BLOCK_NC)
    cols = tl.arange(0, BLOCK_S)
    ptrs = rows[:, None] * S + cols[None, :]
    rmask = rows < NC
    mask = rmask[:, None] & (cols[None, :] < S)
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    inter = tl.sum(x * y, axis=1)
    denom = tl.sum(x * x + y * y, axis=1)
    dc = 2.0 * (inter + smooth) / (denom + smooth)
    valid = rmask
    if not DO_BG:
        valid = valid & ((rows % C) != 0)
    dc = tl.where(valid, dc, 0.0)
    total = tl.sum(dc)
    count = tl.sum(valid.to(tl.float32))
    tl.store(out_ptr, -total / count)


@triton.jit
def _dice_reduce_kernel(x_ptr, y_ptr, inter_ptr, denom_ptr, S, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    base = pid * S
    inter_acc = 0.0
    denom_acc = 0.0
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        inter_acc += tl.sum(x * y)
        denom_acc += tl.sum(x * x + y * y)
    tl.store(inter_ptr + pid, inter_acc)
    tl.store(denom_ptr + pid, denom_acc)


class SoftDiceLossSquaredNew(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0):
        super(SoftDiceLossSquaredNew, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if all([(i == j) for i, j in zip(x.shape, y.shape)]):
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device)
                y_onehot.scatter_(1, y, 1)

        N, C = shp_x[0], shp_x[1]
        S = 1
        for d in shp_x[2:]:
            S *= d

        x = x.contiguous()
        y_onehot = y_onehot.contiguous()

        NC = N * C
        BLOCK_NC = triton.next_power_of_2(NC)
        BLOCK_S = triton.next_power_of_2(S)
        if not self.batch_dice and BLOCK_NC * BLOCK_S <= 16384:
            out = torch.empty((), device=x.device, dtype=torch.float32)
            _dice_fused_kernel[(1,)](x, y_onehot, out, NC, C, S, float(self.smooth),
                                     DO_BG=self.do_bg, BLOCK_NC=BLOCK_NC,
                                     BLOCK_S=BLOCK_S, num_warps=1)
            return out

        inter = torch.empty((N * C,), device=x.device, dtype=torch.float32)
        denom = torch.empty((N * C,), device=x.device, dtype=torch.float32)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(S))
        grid = (N * C,)
        _dice_reduce_kernel[grid](x, y_onehot, inter, denom, S,
                                  BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        inter = inter.view(N, C)
        denom = denom.view(N, C)

        if self.batch_dice:
            inter = inter.sum(0)
            denom = denom.sum(0)

        inter = inter + self.smooth
        denom = denom + self.smooth
        dc = 2 * inter / denom

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc
