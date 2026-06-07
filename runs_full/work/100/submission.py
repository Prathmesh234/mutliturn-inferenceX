import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, IN: tl.constexpr, H: tl.constexpr, OUT: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_IN: tl.constexpr,
                BLOCK_H: tl.constexpr, BLOCK_OUT: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    in_idx = tl.arange(0, BLOCK_IN)
    h_idx = tl.arange(0, BLOCK_H)
    out_idx = tl.arange(0, BLOCK_OUT)
    in_mask = in_idx < IN
    h_mask = h_idx < H
    out_mask = out_idx < OUT

    x = tl.load(x_ptr + offs_m[:, None] * IN + in_idx[None, :],
                mask=mask_m[:, None] & in_mask[None, :], other=0.0)

    w1t = tl.load(w1_ptr + in_idx[:, None] + h_idx[None, :] * IN,
                  mask=in_mask[:, None] & h_mask[None, :], other=0.0)
    b1 = tl.load(b1_ptr + h_idx, mask=h_mask, other=0.0)
    a1 = tl.dot(x, w1t) + b1[None, :]
    a1 = tl.maximum(a1, 0.0)

    w2t = tl.load(w2_ptr + h_idx[:, None] + h_idx[None, :] * H,
                  mask=h_mask[:, None] & h_mask[None, :], other=0.0)
    b2 = tl.load(b2_ptr + h_idx, mask=h_mask, other=0.0)
    a2 = tl.dot(a1, w2t) + b2[None, :]
    a2 = tl.maximum(a2, 0.0)

    w3t = tl.load(w3_ptr + h_idx[:, None] + out_idx[None, :] * H,
                  mask=h_mask[:, None] & out_mask[None, :], other=0.0)
    b3 = tl.load(b3_ptr + out_idx, mask=out_mask, other=0.0)
    out = tl.dot(a2, w3t) + b3[None, :]

    tl.store(out_ptr + offs_m[:, None] * OUT + out_idx[None, :],
             out, mask=mask_m[:, None] & out_mask[None, :])


def _pad16(n):
    p = 1 << (max(n, 1) - 1).bit_length()
    return max(p, 16)


class SmallNNNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, out_channels)
        self.IN = in_channels
        self.H = 32
        self.OUT = out_channels
        self.BIN = _pad16(in_channels)
        self.BH = _pad16(32)
        self.BOUT = _pad16(out_channels)
        self._cache = {}

    def forward(self, xb):
        IN = self.IN
        OUT = self.OUT
        x = xb.reshape(-1, IN)
        M = x.shape[0]
        out = torch.empty((M, OUT), device=x.device, dtype=x.dtype)

        key = xb.shape
        c = self._cache.get(key)
        if c is None:
            BLOCK_M = max(triton.next_power_of_2(M), 16)
            c = (BLOCK_M, (triton.cdiv(M, BLOCK_M),), xb.shape[:-1] + (OUT,))
            self._cache[key] = c
        BLOCK_M, grid, out_shape = c
        _mlp_kernel[grid](
            x, self.l1.weight, self.l1.bias,
            self.l2.weight, self.l2.bias,
            self.l3.weight, self.l3.bias, out,
            M, IN, self.H, OUT,
            BLOCK_M, self.BIN, self.BH, self.BOUT,
            num_warps=1,
        )
        return out.view(out_shape)
