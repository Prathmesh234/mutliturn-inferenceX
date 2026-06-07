import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(feat_ptr, wv_ptr, bv_ptr, wc_ptr, bc_ptr, mask_ptr, out_ptr, w_ptr,
                  B, nseg, idim, odim, nhead, d, scale,
                  HAS_BIAS: tl.constexpr,
                  BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_bh = tl.program_id(0)
    b = pid_bh // nhead
    h = pid_bh % nhead
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    hd = h * d + offs_d

    fmask = (offs_s[:, None] < nseg) & (offs_k[None, :] < idim)
    F = tl.load(feat_ptr + b * nseg * idim + offs_s[:, None] * idim + offs_k[None, :],
                mask=fmask, other=0.0)

    wmask = (offs_d[:, None] < d) & (offs_k[None, :] < idim)
    Wv = tl.load(wv_ptr + hd[:, None] * idim + offs_k[None, :], mask=wmask, other=0.0)
    Wcq = tl.load(wc_ptr + hd[:, None] * idim + offs_k[None, :], mask=wmask, other=0.0)
    Wcv = tl.load(wc_ptr + (odim + hd)[:, None] * idim + offs_k[None, :], mask=wmask, other=0.0)

    mk = tl.dot(F, tl.trans(Wv))
    mq = tl.dot(F, tl.trans(Wcq))
    mv = tl.dot(F, tl.trans(Wcv))
    if HAS_BIAS:
        dmask = offs_d < d
        mk += tl.load(bv_ptr + hd, mask=dmask, other=0.0)[None, :]
        mq += tl.load(bc_ptr + hd, mask=dmask, other=0.0)[None, :]
        mv += tl.load(bc_ptr + odim + hd, mask=dmask, other=0.0)[None, :]

    scores = tl.dot(mk, tl.trans(mq)) * scale

    maskv = tl.load(mask_ptr + b * nseg + offs_s, mask=offs_s < nseg, other=0.0)
    scores = tl.where(maskv[None, :] == 0, -1e9, scores)
    scores = tl.where(offs_s[None, :] < nseg, scores, float("-inf"))

    mx = tl.max(scores, axis=1)
    p = tl.exp(scores - mx[:, None])
    s = tl.sum(p, axis=1)
    w = p / s[:, None]

    r = tl.dot(w, mv)
    featr = tl.load(feat_ptr + b * nseg * idim + offs_s[:, None] * idim + hd[None, :],
                    mask=(offs_s[:, None] < nseg) & (offs_d[None, :] < d), other=0.0)
    out = featr + r
    tl.store(out_ptr + b * nseg * odim + offs_s[:, None] * odim + hd[None, :], out,
             mask=(offs_s[:, None] < nseg) & (offs_d[None, :] < d))
    tl.store(w_ptr + b * nhead * nseg * nseg + h * nseg * nseg
             + offs_s[:, None] * nseg + offs_s[None, :], w,
             mask=(offs_s[:, None] < nseg) & (offs_s[None, :] < nseg))


class MultiHeadAttentionNew(nn.Module):
    def __init__(self, idim, odim, nhead=1, use_bias=True):
        super(MultiHeadAttentionNew, self).__init__()
        self.idim = idim
        self.odim = odim
        self.nheads = nhead
        self.use_bias = use_bias
        self.c_lin = nn.Linear(self.idim, self.odim * 2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0)

    def forward(self, m_feats, mask):
        mask = mask.float().contiguous()
        B, nseg = mask.shape
        m_feats = m_feats.contiguous()
        idim = self.idim
        odim = self.odim
        nhead = self.nheads
        d = odim // nhead

        out = torch.empty((B, nseg, odim), device=m_feats.device, dtype=torch.float32)
        w_out = torch.empty((B, nhead, nseg, nseg), device=m_feats.device, dtype=torch.float32)

        BLOCK_S = max(16, triton.next_power_of_2(nseg))
        BLOCK_D = max(16, triton.next_power_of_2(d))
        BLOCK_K = max(16, triton.next_power_of_2(idim))
        scale = 1.0 / (d ** 0.5)
        has_bias = self.use_bias
        bv = self.v_lin.bias if has_bias else m_feats
        bc = self.c_lin.bias if has_bias else m_feats

        grid = (B * nhead,)
        _fused_kernel[grid](m_feats, self.v_lin.weight, bv, self.c_lin.weight, bc,
                            mask, out, w_out,
                            B, nseg, idim, odim, nhead, d, scale,
                            HAS_BIAS=has_bias,
                            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
                            num_warps=1)
        return out, w_out
