import torch
import torch.nn as nn
import triton
import triton.language as tl


def _np2(x):
    n = 1
    while n < x:
        n *= 2
    return max(16, n)


@triton.jit
def _fused_kernel(lf_ptr, gf_ptr, vw_ptr, vb_ptr, cw_ptr, cb_ptr, ow_ptr,
                  gm_ptr, out_ptr, w_ptr,
                  B, L, G, IDIM, ODIM, DH, N, scale,
                  HAS_BIAS: tl.constexpr, NHEAD: tl.constexpr,
                  BLOCK_L: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr,
                  BLOCK_I: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    offs_l = tl.arange(0, BLOCK_L)
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    offs_i = tl.arange(0, BLOCK_I)
    offs_n = tl.arange(0, BLOCK_N)
    ml = offs_l < L
    mg = offs_g < G
    md = offs_d < DH
    mi = offs_i < IDIM
    mn = offs_n < N
    TWO_OUT = IDIM + ODIM

    lf = tl.load(lf_ptr + b * L * IDIM + offs_l[:, None] * IDIM + offs_i[None, :],
                 mask=ml[:, None] & mi[None, :], other=0.0)
    gf = tl.load(gf_ptr + b * G * IDIM + offs_g[:, None] * IDIM + offs_i[None, :],
                 mask=mg[:, None] & mi[None, :], other=0.0)

    # out_lin contribution from local_feats (columns [0, IDIM) of out weight)
    wl = tl.load(ow_ptr + offs_n[:, None] * TWO_OUT + offs_i[None, :],
                 mask=mn[:, None] & mi[None, :], other=0.0)
    acc = tl.dot(lf, tl.trans(wl))  # [BLOCK_L, BLOCK_N]

    gm = tl.load(gm_ptr + b * G + offs_g, mask=mg, other=0.0)
    valid = mg & (gm != 0.0)

    for h in range(NHEAD):
        drow = h * DH + offs_d
        vw = tl.load(vw_ptr + drow[:, None] * IDIM + offs_i[None, :],
                     mask=md[:, None] & mi[None, :], other=0.0)
        cwq = tl.load(cw_ptr + drow[:, None] * IDIM + offs_i[None, :],
                      mask=md[:, None] & mi[None, :], other=0.0)
        cwv = tl.load(cw_ptr + (ODIM + drow)[:, None] * IDIM + offs_i[None, :],
                      mask=md[:, None] & mi[None, :], other=0.0)
        mk = tl.dot(lf, tl.trans(vw))   # [BLOCK_L, BLOCK_D]
        mq = tl.dot(gf, tl.trans(cwq))  # [BLOCK_G, BLOCK_D]
        mv = tl.dot(gf, tl.trans(cwv))  # [BLOCK_G, BLOCK_D]
        if HAS_BIAS:
            mk += tl.load(vb_ptr + drow, mask=md, other=0.0)[None, :]
            mq += tl.load(cb_ptr + drow, mask=md, other=0.0)[None, :]
            mv += tl.load(cb_ptr + ODIM + drow, mask=md, other=0.0)[None, :]

        scores = tl.dot(mk, tl.trans(mq)) * scale  # [BLOCK_L, BLOCK_G]
        scores = tl.where(valid[None, :], scores, -1e9)
        m = tl.max(scores, axis=1)
        p = tl.exp(scores - m[:, None])
        s = tl.sum(p, axis=1)
        probs = p / s[:, None]

        tl.store(w_ptr + b * NHEAD * L * G + h * L * G + offs_l[:, None] * G + offs_g[None, :],
                 probs, mask=ml[:, None] & mg[None, :])

        r_head = tl.dot(probs, mv)  # [BLOCK_L, BLOCK_D]
        # out_lin contribution from this head (columns [IDIM+h*DH, IDIM+h*DH+DH))
        wr = tl.load(ow_ptr + offs_n[:, None] * TWO_OUT + (IDIM + drow)[None, :],
                     mask=mn[:, None] & md[None, :], other=0.0)
        acc += tl.dot(r_head, tl.trans(wr))

    tl.store(out_ptr + b * L * N + offs_l[:, None] * N + offs_n[None, :], acc,
             mask=ml[:, None] & mn[None, :])


class MutiLevelEnhanceNew(nn.Module):
    def __init__(self, idim, odim, nhead=1, use_bias=True):
        super(MutiLevelEnhanceNew, self).__init__()
        self.idim = idim
        self.odim = odim
        self.nheads = nhead
        self.use_bias = use_bias
        self.c_lin = nn.Linear(self.idim, self.odim * 2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0)
        self.out_lin = nn.Linear(2 * self.odim, self.odim, bias=False)

    def forward(self, local_feats, global_feats, local_mask, global_mask):
        B, L, _ = local_feats.shape
        G = global_feats.shape[1]
        odim = self.odim
        nhead = self.nheads
        dh = odim // nhead

        lf = local_feats.contiguous()
        gf = global_feats.contiguous()
        gmask = global_mask.float().contiguous()

        out = torch.empty((B, L, odim), device=lf.device, dtype=torch.float32)
        w = torch.empty((B, nhead, L, G), device=lf.device, dtype=torch.float32)

        scale = 1.0 / (dh ** 0.5)
        vb = self.v_lin.bias if self.use_bias else lf
        cb = self.c_lin.bias if self.use_bias else lf
        _fused_kernel[(B,)](lf, gf, self.v_lin.weight, vb, self.c_lin.weight, cb,
                            self.out_lin.weight, gmask, out, w,
                            B, L, G, self.idim, odim, dh, odim, scale,
                            self.use_bias, nhead,
                            BLOCK_L=_np2(L), BLOCK_G=_np2(G), BLOCK_D=_np2(dh),
                            BLOCK_I=_np2(self.idim), BLOCK_N=_np2(odim), num_warps=1)
        return out, w
