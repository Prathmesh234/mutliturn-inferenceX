import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _fused_all(x_ptr, w_ptr, b_ptr, t_ptr, out_ptr,
               G, C2, B, H, N, num_groups,
               sxa, sxb, sxc, sxd, sw0,
               sta, stb, stc, ste,
               BG: tl.constexpr, BB: tl.constexpr,
               BH: tl.constexpr, BN: tl.constexpr):
    offs_g = tl.arange(0, BG)
    offs_b = tl.arange(0, BB)
    offs_h = tl.arange(0, BH)
    offs_n = tl.arange(0, BN)
    mask_g = offs_g < G
    mask_b = offs_b < B
    mask_h = offs_h < H
    mask_n = offs_n < N
    a_idx = offs_g // C2
    c_idx = offs_g % C2
    gbase_x = a_idx * sxa + c_idx * sxc          # (BG,)
    gbase_t = a_idx * sta + c_idx * stc          # (BG,)
    # x (BG,BB,BH)
    x = tl.load(x_ptr + gbase_x[:, None, None]
                + offs_b[None, :, None] * sxb
                + offs_h[None, None, :] * sxd,
                mask=mask_g[:, None, None] & mask_b[None, :, None] & mask_h[None, None, :],
                other=0.0).to(tl.float32)
    # w (BN,BH)
    w = tl.load(w_ptr + offs_n[:, None] * sw0 + offs_h[None, :],
                mask=mask_n[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    # logits (BG,BB,BN)
    logit = tl.sum(x[:, :, None, :] * w[None, None, :, :], axis=3) + bias[None, None, :]
    valid = mask_g[:, None, None] & mask_b[None, :, None] & mask_n[None, None, :]
    logit = tl.where(valid, logit, -float('inf'))
    m = tl.max(logit, axis=1)                    # (BG,BN)
    ex = tl.exp(logit - m[:, None, :])
    s = tl.sum(ex, axis=1)                        # (BG,BN)
    lsm = logit - m[:, None, :] - tl.log(s)[:, None, :]
    tgt = tl.load(t_ptr + gbase_t[:, None, None]
                  + offs_b[None, :, None] * stb
                  + offs_n[None, None, :] * ste,
                  mask=valid, other=0.0).to(tl.float32)
    contrib = tl.where(valid, tgt * lsm, 0.0)
    total = -tl.sum(contrib)
    tl.store(out_ptr, total / num_groups)


class SoftmaxLossNew(nn.Module):
    def __init__(self, hidden_dim, speaker_num, **kwargs):
        super(SoftmaxLossNew, self).__init__()
        self.fc = nn.Linear(hidden_dim, speaker_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_BxH, labels_B):
        H = self.fc.in_features
        N = self.fc.out_features
        a, B, C2, _ = x_BxH.shape
        G = a * C2
        num_groups = G * N
        sxa, sxb, sxc, sxd = x_BxH.stride()
        sta, stb, stc, ste = labels_B.stride()
        sw0 = self.fc.weight.stride(0)
        out = torch.empty((1,), device=x_BxH.device, dtype=torch.float32)
        t = labels_B if labels_B.dtype == torch.float32 else labels_B.float()
        _fused_all[(1,)](x_BxH, self.fc.weight, self.fc.bias, t, out,
                         G, C2, B, H, N, num_groups,
                         sxa, sxb, sxc, sxd, sw0,
                         sta, stb, stc, ste,
                         BG=_next_pow2(G), BB=_next_pow2(B),
                         BH=_next_pow2(H), BN=_next_pow2(N),
                         num_warps=4)
        return out[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'speaker_num': 4}]
