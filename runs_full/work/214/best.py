import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(h_ptr, wq_ptr, wkv_ptr, wo_ptr, lw_ptr, lb_ptr, o_ptr,
                 L, B, D, Dh, scale, eps,
                 H: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr,
                 BDH: tl.constexpr):
    b = tl.program_id(0)
    HDh = H * Dh
    offs_l = tl.arange(0, BL)
    offs_dh = tl.arange(0, BDH)
    offd = tl.arange(0, BD)
    mask_l = offs_l < L
    mask_dh = offs_dh < Dh
    mask_d = offd < D
    hb_off = b * D + offs_l[:, None] * (B * D) + offd[None, :]
    h_b = tl.load(h_ptr + hb_off, mask=mask_l[:, None] & mask_d[None, :], other=0.0)
    acc = tl.zeros((BL, BD), tl.float32)
    wmask = mask_dh[:, None] & mask_d[None, :]
    for n in range(H):
        wrow = (n * Dh + offs_dh)[:, None] * D + offd[None, :]
        wq = tl.load(wq_ptr + wrow, mask=wmask, other=0.0)
        wk = tl.load(wkv_ptr + wrow, mask=wmask, other=0.0)
        wv = tl.load(wkv_ptr + HDh * D + wrow, mask=wmask, other=0.0)
        q = tl.dot(h_b, tl.trans(wq), allow_tf32=False)
        k = tl.dot(h_b, tl.trans(wk), allow_tf32=False)
        v = tl.dot(h_b, tl.trans(wv), allow_tf32=False)
        scores = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
        scores = tl.where(mask_l[None, :], scores, -float('inf'))
        m = tl.max(scores, axis=1)
        p = tl.exp(scores - m[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        av = tl.dot(p, v, allow_tf32=False)
        wo = tl.load(wo_ptr + offd[:, None] * HDh + (n * Dh + offs_dh)[None, :],
                     mask=mask_d[:, None] & mask_dh[None, :], other=0.0)
        acc += tl.dot(av, tl.trans(wo), allow_tf32=False)
    val = acc + h_b
    mean = tl.sum(tl.where(mask_d[None, :], val, 0.0), axis=1) / D
    vc = tl.where(mask_d[None, :], val - mean[:, None], 0.0)
    var = tl.sum(vc * vc, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    lw = tl.load(lw_ptr + offd, mask=mask_d, other=0.0)
    lb = tl.load(lb_ptr + offd, mask=mask_d, other=0.0)
    out = vc * rstd[:, None] * lw[None, :] + lb[None, :]
    tl.store(o_ptr + hb_off, out, mask=mask_l[:, None] & mask_d[None, :])


def _np2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class MultiHeadAttnNew(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        H, Dh, D = self.n_head, self.d_head, self.d_model
        L, Bsz = h.size(0), h.size(1)
        h_flat = h.reshape(L * Bsz, D).contiguous().float()
        Wq = self.q_net.weight.contiguous()
        Wkv = self.kv_net.weight.contiguous()
        Wo = self.o_net.weight.contiguous()
        out = torch.empty_like(h_flat)
        BL = max(16, _np2(L))
        BD = max(16, _np2(D))
        BDH = max(16, _np2(Dh))
        fused_kernel[(Bsz,)](h_flat, Wq, Wkv, Wo, self.layer_norm.weight,
                             self.layer_norm.bias, out, L, Bsz, D, Dh,
                             self.scale, 1e-5, H=H, BL=BL, BD=BD, BDH=BDH,
                             num_warps=1)
        return out.reshape(L, Bsz, D).to(h.dtype)
