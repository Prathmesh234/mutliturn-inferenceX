import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _joiner_kernel(
    x_ptr, y_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    B, Dx, Dy, H, D,
    BLOCK_H: tl.constexpr, BLOCK_DX: tl.constexpr, BLOCK_DY: tl.constexpr,
):
    row = tl.program_id(0)

    offs_h = tl.arange(0, BLOCK_H)
    offs_dx = tl.arange(0, BLOCK_DX)
    offs_dy = tl.arange(0, BLOCK_DY)
    mask_h = offs_h < H
    mask_dx = offs_dx < Dx
    mask_dy = offs_dy < Dy

    # load x and y rows
    x = tl.load(x_ptr + row * Dx + offs_dx, mask=mask_dx, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * Dy + offs_dy, mask=mask_dy, other=0.0).to(tl.float32)

    # W1: [H, D], columns [0:Dx] correspond to x, [Dx:D] to y
    w1x = tl.load(w1_ptr + offs_h[:, None] * D + offs_dx[None, :],
                  mask=mask_h[:, None] & mask_dx[None, :], other=0.0).to(tl.float32)
    w1y = tl.load(w1_ptr + offs_h[:, None] * D + (Dx + offs_dy[None, :]),
                  mask=mask_h[:, None] & mask_dy[None, :], other=0.0).to(tl.float32)

    b1 = tl.load(b1_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    hidden = tl.sum(w1x * x[None, :], axis=1) + tl.sum(w1y * y[None, :], axis=1) + b1
    hidden = tl.where(hidden > 0.0, hidden, 0.0)
    hidden = tl.where(mask_h, hidden, 0.0)

    # W2: [1, H]
    w2 = tl.load(w2_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b2 = tl.load(b2_ptr).to(tl.float32)

    out = tl.sum(hidden * w2, axis=0) + b2
    tl.store(out_ptr + row, out)


class JoinerNew(nn.Module):

    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_latent_dim + y_latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        B, Dx = x.shape
        Dy = y.shape[1]
        H = self.fc1.weight.shape[0]
        D = Dx + Dy
        out = torch.empty((B, 1), device=x.device, dtype=x.dtype)

        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_DX = triton.next_power_of_2(Dx)
        BLOCK_DY = triton.next_power_of_2(Dy)

        _joiner_kernel[(B,)](
            x, y, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias, out,
            B, Dx, Dy, H, D,
            BLOCK_H=BLOCK_H, BLOCK_DX=BLOCK_DX, BLOCK_DY=BLOCK_DY,
            num_warps=4,
        )
        return out
