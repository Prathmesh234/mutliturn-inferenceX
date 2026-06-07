import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bva_kernel(
    x_ptr, wt_ptr, bt_ptr, wph_ptr, bph_ptr, wpsi_ptr, bpsi_ptr,
    wr1_ptr, br1_ptr, wr2_ptr, br2_ptr, out_ptr,
    N, L, C, H, C2,
    BN: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, BC2: tl.constexpr,
):
    offs_n = tl.arange(0, BN)
    offs_l = tl.arange(0, BL)
    offs_c = tl.arange(0, BC)
    offs_h = tl.arange(0, BH)
    offs_c2 = tl.arange(0, BC2)

    mask_n = offs_n < N
    mask_l = offs_l < L
    mask_c = offs_c < C
    mask_h = offs_h < H
    mask_c2 = offs_c2 < C2

    # x: [BN, BL, BC]
    x = tl.load(x_ptr + offs_n[:, None, None] * L * C + offs_l[None, :, None] * C + offs_c[None, None, :],
                mask=mask_n[:, None, None] & mask_l[None, :, None] & mask_c[None, None, :], other=0.0)

    wt = tl.load(wt_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    wph = tl.load(wph_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    wpsi = tl.load(wpsi_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    bt = tl.load(bt_ptr + offs_h, mask=mask_h, other=0.0)
    bph = tl.load(bph_ptr + offs_h, mask=mask_h, other=0.0)
    bpsi = tl.load(bpsi_ptr + offs_h, mask=mask_h, other=0.0)

    # x_t[n,l,h] = sum_c x[n,l,c]*wt[h,c] -> [BN, BL, BH]
    x_t = tl.sum(x[:, :, None, :] * wt[None, None, :, :], axis=3) + bt[None, None, :]
    x_ph = tl.sum(x[:, :, None, :] * wph[None, None, :, :], axis=3) + bph[None, None, :]
    x_psi = tl.sum(x[:, :, None, :] * wpsi[None, None, :, :], axis=3) + bpsi[None, None, :]

    # attn[n,l,m] = sum_h x_ph[n,l,h]*x_t[n,m,h] -> [BN, BL, BL]
    attn = tl.sum(x_ph[:, :, None, :] * x_t[:, None, :, :], axis=3)
    attn = tl.where(mask_l[None, None, :], attn, float("-inf"))
    attn = attn - tl.max(attn, axis=2)[:, :, None]
    attn = tl.exp(attn)
    attn = attn / tl.sum(attn, axis=2)[:, :, None]

    # x_add[n,l,h] = sum_m attn[n,l,m]*x_psi[n,m,h] -> [BN, BL, BH]
    x_add = tl.sum(attn[:, :, :, None] * x_psi[:, None, :, :], axis=2)

    wr1 = tl.load(wr1_ptr + offs_c2[:, None] * H + offs_h[None, :], mask=mask_c2[:, None] & mask_h[None, :], other=0.0)
    br1 = tl.load(br1_ptr + offs_c2, mask=mask_c2, other=0.0)
    r1 = tl.sum(x_add[:, :, None, :] * wr1[None, None, :, :], axis=3) + br1[None, None, :]
    r1 = tl.where(r1 > 0, r1, 0.2 * r1)

    wr2 = tl.load(wr2_ptr + offs_c[:, None] * C2 + offs_c2[None, :], mask=mask_c[:, None] & mask_c2[None, :], other=0.0)
    br2 = tl.load(br2_ptr + offs_c, mask=mask_c, other=0.0)
    r2 = tl.sum(r1[:, :, None, :] * wr2[None, None, :, :], axis=3) + br2[None, None, :]
    e = tl.exp(2 * r2)
    r2 = (e - 1) / (e + 1)

    out = x + r2
    tl.store(out_ptr + offs_n[:, None, None] * L * C + offs_l[None, :, None] * C + offs_c[None, None, :],
             out, mask=mask_n[:, None, None] & mask_l[None, :, None] & mask_c[None, None, :])


def _next_pow2(x):
    return 1 << (max(x, 1) - 1).bit_length()


class BatchedVectorAttentionNew(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(input_dim, hidden_dim)
        self.psi = nn.Linear(input_dim, hidden_dim)
        self.recover1 = nn.Linear(hidden_dim, max(input_dim // 2, 1))
        self.lrelu = nn.LeakyReLU(0.2)
        self.recover2 = nn.Linear(max(input_dim // 2, 1), input_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        n, L, C = x.shape
        H = self.theta.weight.shape[0]
        C2 = self.recover1.weight.shape[0]
        x = x.contiguous()
        out = torch.empty_like(x)
        _bva_kernel[(1,)](
            x, self.theta.weight, self.theta.bias,
            self.phi.weight, self.phi.bias,
            self.psi.weight, self.psi.bias,
            self.recover1.weight, self.recover1.bias,
            self.recover2.weight, self.recover2.bias,
            out, n, L, C, H, C2,
            BN=_next_pow2(n), BL=_next_pow2(L), BC=_next_pow2(C),
            BH=_next_pow2(H), BC2=_next_pow2(C2),
            num_warps=1,
        )
        return out
