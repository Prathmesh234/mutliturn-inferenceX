import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ffn_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, out_ptr,
                M, D, F,
                sx_m, sx_d,
                sw1_d, sw1_f,
                sw2_f, sw2_d,
                so_m, so_d,
                BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_f = pid * BLOCK_F + tl.arange(0, BLOCK_F)
    mmask = offs_m < M
    dmask = offs_d < D
    fmask = offs_f < F

    x = tl.load(x_ptr + offs_m[:, None] * sx_m + offs_d[None, :] * sx_d,
                mask=mmask[:, None] & dmask[None, :], other=0.0)
    w1 = tl.load(w1_ptr + offs_d[:, None] * sw1_d + offs_f[None, :] * sw1_f,
                 mask=dmask[:, None] & fmask[None, :], other=0.0)
    h = tl.dot(x, w1, allow_tf32=False)
    b1 = tl.load(b1_ptr + offs_f, mask=fmask, other=0.0)
    h = tl.maximum(h + b1[None, :], 0.0)
    w2 = tl.load(w2_ptr + offs_f[:, None] * sw2_f + offs_d[None, :] * sw2_d,
                 mask=fmask[:, None] & dmask[None, :], other=0.0)
    part = tl.dot(h, w2, allow_tf32=False)
    tl.atomic_add(out_ptr + offs_m[:, None] * so_m + offs_d[None, :] * so_d,
                  part, mask=mmask[:, None] & dmask[None, :])


class FeedForwardNew(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        orig_shape = x.shape
        D = orig_shape[-1]
        x2 = x.reshape(-1, D).contiguous()
        M = x2.shape[0]
        F = self.linear_1.weight.shape[0]
        out = self.linear_2.bias.to(torch.float32).unsqueeze(0).expand(M, D).contiguous()
        w1 = self.linear_1.weight
        w2 = self.linear_2.weight
        BLOCK_M = max(16, triton.next_power_of_2(M))
        BLOCK_D = max(16, triton.next_power_of_2(D))
        BLOCK_F = 64
        grid = (triton.cdiv(F, BLOCK_F),)
        _ffn_kernel[grid](x2, w1, self.linear_1.bias, w2, out,
                          M, D, F,
                          x2.stride(0), x2.stride(1),
                          w1.stride(1), w1.stride(0),
                          w2.stride(1), w2.stride(0),
                          out.stride(0), out.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_F=BLOCK_F,
                          num_warps=4, num_stages=1)
        return out.reshape(*orig_shape[:-1], D)
