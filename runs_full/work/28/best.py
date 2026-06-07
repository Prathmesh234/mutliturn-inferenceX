import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ffn_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                lnw_ptr, lnb_ptr, out_ptr,
                M, d_in, d_hid, eps,
                BM: tl.constexpr, DIN: tl.constexpr, DHID: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M

    di = tl.arange(0, DIN)
    dj = tl.arange(0, DHID)
    in_mask = di < d_in
    hid_mask = dj < d_hid

    x = tl.load(x_ptr + rows[:, None] * d_in + di[None, :],
                mask=rmask[:, None] & in_mask[None, :], other=0.0)

    w1 = tl.load(w1_ptr + dj[:, None] * d_in + di[None, :],
                 mask=hid_mask[:, None] & in_mask[None, :], other=0.0)
    b1 = tl.load(b1_ptr + dj, mask=hid_mask, other=0.0)

    h = tl.sum(x[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
    h = tl.maximum(h, 0.0)

    w2 = tl.load(w2_ptr + di[:, None] * d_hid + dj[None, :],
                 mask=in_mask[:, None] & hid_mask[None, :], other=0.0)
    b2 = tl.load(b2_ptr + di, mask=in_mask, other=0.0)

    y = tl.sum(h[:, None, :] * w2[None, :, :], axis=2) + b2[None, :]
    y = y + x

    cnt = d_in
    ymask = rmask[:, None] & in_mask[None, :]
    mean = tl.sum(tl.where(ymask, y, 0.0), axis=1) / cnt
    yc = tl.where(ymask, y - mean[:, None], 0.0)
    var = tl.sum(yc * yc, axis=1) / cnt
    rstd = 1.0 / tl.sqrt(var + eps)
    yn = yc * rstd[:, None]

    lnw = tl.load(lnw_ptr + di, mask=in_mask, other=0.0)
    lnb = tl.load(lnb_ptr + di, mask=in_mask, other=0.0)
    out = yn * lnw[None, :] + lnb[None, :]

    tl.store(out_ptr + rows[:, None] * d_in + di[None, :], out, mask=ymask)


class PositionwiseFeedForwardNew(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        d_in = self.w_1.in_features
        d_hid = self.w_1.out_features
        orig_shape = x.shape
        xf = x.reshape(-1, d_in).contiguous()
        M = xf.shape[0]
        out = torch.empty_like(xf)

        DIN = triton.next_power_of_2(d_in)
        DHID = triton.next_power_of_2(d_hid)
        BM = 64
        grid = (triton.cdiv(M, BM),)
        _ffn_kernel[grid](
            xf, self.w_1.weight, self.w_1.bias,
            self.w_2.weight, self.w_2.bias,
            self.layer_norm.weight, self.layer_norm.bias, out,
            M, d_in, d_hid, self.layer_norm.eps,
            BM=BM, DIN=DIN, DHID=DHID, num_warps=4,
        )
        return out.reshape(orig_shape)
