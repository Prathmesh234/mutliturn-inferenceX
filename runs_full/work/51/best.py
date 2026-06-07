import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ffn_kernel(x_ptr, ln_w_ptr, ln_b_ptr,
                w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                N, D: tl.constexpr, F: tl.constexpr,
                BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
                eps):
    row = tl.program_id(0)
    if row >= N:
        return
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_f = tl.arange(0, BLOCK_F)
    mask_f = offs_f < F

    x = tl.load(x_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask_d, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(ln_w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    b = tl.load(ln_b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    ln = xc * rstd * g + b

    w1 = tl.load(w1_ptr + offs_f[:, None] * D + offs_d[None, :],
                 mask=mask_f[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    inter = tl.sum(ln[None, :] * w1, axis=1)
    b1 = tl.load(b1_ptr + offs_f, mask=mask_f, other=0.0).to(tl.float32)
    inter = inter + b1
    inter = tl.where(inter > 0.0, inter, 0.0)
    inter = tl.where(mask_f, inter, 0.0)

    w2 = tl.load(w2_ptr + offs_d[:, None] * F + offs_f[None, :],
                 mask=mask_d[:, None] & mask_f[None, :], other=0.0).to(tl.float32)
    out = tl.sum(inter[None, :] * w2, axis=1)
    b2 = tl.load(b2_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    out = out + b2 + x

    tl.store(out_ptr + row * D + offs_d, out, mask=mask_d)


class PositionwiseFeedForwardNew(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardNew, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x):
        orig_shape = x.shape
        D = self.d_model
        F = self.d_ff
        x2d = x.reshape(-1, D).contiguous()
        N = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_F = triton.next_power_of_2(F)
        grid = (N,)
        _ffn_kernel[grid](
            x2d, self.layer_norm.weight, self.layer_norm.bias,
            self.w_1.weight, self.w_1.bias,
            self.w_2.weight, self.w_2.bias, out,
            N, D, F, BLOCK_D, BLOCK_F, 1e-06,
            num_warps=1,
        )
        return out.reshape(orig_shape)
