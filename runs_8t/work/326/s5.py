import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bva_kernel(
    x_ptr, wt_ptr, bt_ptr, wph_ptr, bph_ptr, wpsi_ptr, bpsi_ptr,
    wr1_ptr, br1_ptr, wr2_ptr, br2_ptr, out_ptr,
    L, C, H, C2,
    BL: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, BC2: tl.constexpr,
):
    n = tl.program_id(0)
    row = tl.program_id(1)  # which query row l
    if row >= L:
        return

    offs_l = tl.arange(0, BL)
    offs_c = tl.arange(0, BC)
    offs_h = tl.arange(0, BH)
    offs_c2 = tl.arange(0, BC2)

    mask_l = offs_l < L
    mask_c = offs_c < C
    mask_h = offs_h < H
    mask_c2 = offs_c2 < C2

    base = n * L * C
    # full x[n]: [BL, BC]
    x = tl.load(x_ptr + base + offs_l[:, None] * C + offs_c[None, :],
                mask=mask_l[:, None] & mask_c[None, :], other=0.0)
    # single query row x[n,row]: [BC]
    xq = tl.load(x_ptr + base + row * C + offs_c, mask=mask_c, other=0.0)

    wt = tl.load(wt_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    wph = tl.load(wph_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    wpsi = tl.load(wpsi_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    bt = tl.load(bt_ptr + offs_h, mask=mask_h, other=0.0)
    bph = tl.load(bph_ptr + offs_h, mask=mask_h, other=0.0)
    bpsi = tl.load(bpsi_ptr + offs_h, mask=mask_h, other=0.0)

    # x_t[m,h] for all m, x_psi[m,h] for all m  -> [BL, BH]
    x_t = tl.sum(x[:, None, :] * wt[None, :, :], axis=2) + bt[None, :]
    x_psi = tl.sum(x[:, None, :] * wpsi[None, :, :], axis=2) + bpsi[None, :]
    # x_ph for query row only -> [BH]
    x_ph = tl.sum(xq[None, :] * wph, axis=1) + bph

    # attn[m] = sum_h x_ph[h]*x_t[m,h] -> [BL]
    attn = tl.sum(x_ph[None, :] * x_t, axis=1)
    attn = tl.where(mask_l, attn, float("-inf"))
    attn = tl.exp(attn - tl.max(attn, axis=0))
    attn = attn / tl.sum(attn, axis=0)

    # x_add[h] = sum_m attn[m]*x_psi[m,h] -> [BH]
    x_add = tl.sum(attn[:, None] * x_psi, axis=0)

    wr1 = tl.load(wr1_ptr + offs_c2[:, None] * H + offs_h[None, :], mask=mask_c2[:, None] & mask_h[None, :], other=0.0)
    br1 = tl.load(br1_ptr + offs_c2, mask=mask_c2, other=0.0)
    r1 = tl.sum(x_add[None, :] * wr1, axis=1) + br1  # [BC2]
    r1 = tl.where(r1 > 0, r1, 0.2 * r1)

    wr2 = tl.load(wr2_ptr + offs_c[:, None] * C2 + offs_c2[None, :], mask=mask_c[:, None] & mask_c2[None, :], other=0.0)
    br2 = tl.load(br2_ptr + offs_c, mask=mask_c, other=0.0)
    r2 = tl.sum(r1[None, :] * wr2, axis=1) + br2  # [BC]
    e = tl.exp(2 * r2)
    r2 = (e - 1) / (e + 1)

    out = xq + r2
    tl.store(out_ptr + base + row * C + offs_c, out, mask=mask_c)


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
        _bva_kernel[(n, L)](
            x, self.theta.weight, self.theta.bias,
            self.phi.weight, self.phi.bias,
            self.psi.weight, self.psi.bias,
            self.recover1.weight, self.recover1.bias,
            self.recover2.weight, self.recover2.bias,
            out, L, C, H, C2,
            BL=_next_pow2(L), BC=_next_pow2(C), BH=_next_pow2(H), BC2=_next_pow2(C2),
            num_warps=1,
        )
        return out
