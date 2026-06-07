import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _adv_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, wh_ptr, bh_ptr, out_ptr,
                M, DIN: tl.constexpr, H: tl.constexpr, SLOPE: tl.constexpr,
                BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    offk = tl.arange(0, DIN)
    offj = tl.arange(0, H)
    x = tl.load(x_ptr + rows[:, None] * DIN + offk[None, :], mask=rmask[:, None], other=0.0)
    w1 = tl.load(w1_ptr + offj[:, None] * DIN + offk[None, :])
    b1 = tl.load(b1_ptr + offj)
    h1 = tl.sum(x[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
    a1 = tl.where(h1 >= 0, h1, h1 * SLOPE)
    w2 = tl.load(w2_ptr + offj[:, None] * H + offj[None, :])
    b2 = tl.load(b2_ptr + offj)
    h2 = tl.dot(a1, tl.trans(w2)) + b2[None, :]
    a2 = tl.where(h2 >= 0, h2, h2 * SLOPE)
    wh = tl.load(wh_ptr + offj)
    out = tl.sum(a2 * wh[None, :], axis=1) + tl.load(bh_ptr)
    tl.store(out_ptr + rows, out, mask=rmask)


class AdvNew(nn.Module):
    def __init__(self, dim_inputs, dropout):
        super(AdvNew, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 32)
        self.affine2 = nn.Linear(32, 32)
        self.adv_head = nn.Linear(32, 1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        din = self.affine1.in_features
        xf = x.reshape(-1, din).contiguous()
        M = xf.shape[0]
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _adv_kernel[grid](xf, self.affine1.weight, self.affine1.bias,
                          self.affine2.weight, self.affine2.bias,
                          self.adv_head.weight, self.adv_head.bias, out,
                          M, DIN=din, H=32, SLOPE=0.01, BLOCK_M=BLOCK_M, num_warps=1)
        return out.reshape(*x.shape[:-1], 1)
