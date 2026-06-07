import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 << (max(1, x) - 1).bit_length()


@triton.jit
def _fused_attn_fc_ln(q_ptr, k_ptr, v_ptr, mask_ptr, wfc_ptr, res_ptr,
                      g_ptr, b_ptr, y_ptr, attn_ptr, temperature, Dm, eps,
                      H: tl.constexpr, Lq, Lk, Dk: tl.constexpr, Dv: tl.constexpr,
                      HAS_MASK: tl.constexpr, BLOCK_LK: tl.constexpr,
                      BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    row = tl.program_id(0)            # over B*Lq
    b = row // Lq
    qi = row % Lq

    offs_lk = tl.arange(0, BLOCK_LK)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_d = tl.arange(0, BLOCK_D)
    mlk = offs_lk < Lk
    mdk = offs_dk < Dk
    mdv = offs_dv < Dv
    md = offs_d < Dm

    K = H * Dv
    proj = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if HAS_MASK:
        m = tl.load(mask_ptr + b * Lk + offs_lk, mask=mlk, other=0.0)

    for h in range(H):
        q_base = ((b * Lq + qi) * H + h) * Dk
        q = tl.load(q_ptr + q_base + offs_dk, mask=mdk, other=0.0)
        k_off = ((b * Lk + offs_lk) * H + h) * Dk
        k = tl.load(k_ptr + k_off[:, None] + offs_dk[None, :],
                    mask=mlk[:, None] & mdk[None, :], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) / temperature
        if HAS_MASK:
            scores = tl.where(m == 0, -1000000000.0, scores)
        scores = tl.where(mlk, scores, -float('inf'))
        smax = tl.max(scores, axis=0)
        p = tl.exp(scores - smax)
        p = p / tl.sum(p, axis=0)
        tl.store(attn_ptr + ((b * H + h) * Lq + qi) * Lk + offs_lk, p, mask=mlk)

        v_off = ((b * Lk + offs_lk) * H + h) * Dv
        v = tl.load(v_ptr + v_off[:, None] + offs_dv[None, :],
                    mask=mlk[:, None] & mdv[None, :], other=0.0)
        hout = tl.sum(p[:, None] * v, axis=0)  # [Dv]

        wcol = h * Dv + offs_dv
        wtile = tl.load(wfc_ptr + offs_d[:, None] * K + wcol[None, :],
                        mask=md[:, None] & mdv[None, :], other=0.0)
        proj += tl.sum(wtile * hout[None, :], axis=1)

    r = tl.load(res_ptr + row * Dm + offs_d, mask=md, other=0.0)
    xv = proj + r
    mean = tl.sum(tl.where(md, xv, 0.0), axis=0) / Dm
    xc = tl.where(md, xv - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Dm
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + offs_d, mask=md, other=0.0)
    bb = tl.load(b_ptr + offs_d, mask=md, other=0.0)
    y = xc * rstd * g + bb
    tl.store(y_ptr + row * Dm + offs_d, y, mask=md)


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr, attn_ptr,
                 temperature, H, Lq, Lk, Dk, Dv,
                 HAS_MASK: tl.constexpr,
                 BLOCK_LK: tl.constexpr, BLOCK_DK: tl.constexpr,
                 BLOCK_DV: tl.constexpr):
    pid = tl.program_id(0)            # over B*H*Lq
    bh = pid // Lq
    qi = pid % Lq
    b = bh // H
    h = bh % H

    offs_lk = tl.arange(0, BLOCK_LK)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)

    mask_dk = offs_dk < Dk
    mask_lk = offs_lk < Lk
    mask_dv = offs_dv < Dv

    # q row [Dk] from layout [B, Lq, H, Dk]
    q_base = ((b * Lq + qi) * H + h) * Dk
    q = tl.load(q_ptr + q_base + offs_dk, mask=mask_dk, other=0.0)

    # k tile [Lk, Dk] from layout [B, Lk, H, Dk]
    k_off = ((b * Lk + offs_lk) * H + h) * Dk
    k = tl.load(k_ptr + k_off[:, None] + offs_dk[None, :],
                mask=mask_lk[:, None] & mask_dk[None, :], other=0.0)

    scores = tl.sum(q[None, :] * k, axis=1) / temperature  # [Lk]

    if HAS_MASK:
        m = tl.load(mask_ptr + b * Lk + offs_lk, mask=mask_lk, other=0.0)
        scores = tl.where(m == 0, -1000000000.0, scores)

    scores = tl.where(mask_lk, scores, -float('inf'))
    smax = tl.max(scores, axis=0)
    p = tl.exp(scores - smax)
    denom = tl.sum(p, axis=0)
    p = p / denom

    tl.store(attn_ptr + pid * Lk + offs_lk, p, mask=mask_lk)

    # v tile [Lk, Dv] from layout [B, Lk, H, Dv]
    v_off = ((b * Lk + offs_lk) * H + h) * Dv
    v = tl.load(v_ptr + v_off[:, None] + offs_dv[None, :],
                mask=mask_lk[:, None] & mask_dv[None, :], other=0.0)
    out = tl.sum(p[:, None] * v, axis=0)  # [Dv]
    # store to layout [B, Lq, H, Dv]
    o_base = ((b * Lq + qi) * H + h) * Dv
    tl.store(out_ptr + o_base + offs_dv, out, mask=mask_dv)


@triton.jit
def _qkv_kernel(q_ptr, k_ptr, v_ptr, wq_ptr, wk_ptr, wv_ptr,
                qo_ptr, ko_ptr, vo_ptr, Dm, OK, OV,
                BLOCK_DM: tl.constexpr, BLOCK_OK: tl.constexpr,
                BLOCK_OV: tl.constexpr):
    row = tl.program_id(0)
    offs_dm = tl.arange(0, BLOCK_DM)
    mdm = offs_dm < Dm
    xq = tl.load(q_ptr + row * Dm + offs_dm, mask=mdm, other=0.0)
    xk = tl.load(k_ptr + row * Dm + offs_dm, mask=mdm, other=0.0)
    xv = tl.load(v_ptr + row * Dm + offs_dm, mask=mdm, other=0.0)

    offs_ok = tl.arange(0, BLOCK_OK)
    mok = offs_ok < OK
    wq = tl.load(wq_ptr + offs_ok[:, None] * Dm + offs_dm[None, :],
                 mask=mok[:, None] & mdm[None, :], other=0.0)
    wk = tl.load(wk_ptr + offs_ok[:, None] * Dm + offs_dm[None, :],
                 mask=mok[:, None] & mdm[None, :], other=0.0)
    pq = tl.sum(wq * xq[None, :], axis=1)
    pk = tl.sum(wk * xk[None, :], axis=1)
    tl.store(qo_ptr + row * OK + offs_ok, pq, mask=mok)
    tl.store(ko_ptr + row * OK + offs_ok, pk, mask=mok)

    offs_ov = tl.arange(0, BLOCK_OV)
    mov = offs_ov < OV
    wv = tl.load(wv_ptr + offs_ov[:, None] * Dm + offs_dm[None, :],
                 mask=mov[:, None] & mdm[None, :], other=0.0)
    pv = tl.sum(wv * xv[None, :], axis=1)
    tl.store(vo_ptr + row * OV + offs_ov, pv, mask=mov)


@triton.jit
def _fc_ln_kernel(x_ptr, w_ptr, res_ptr, g_ptr, b_ptr, out_ptr, D, K, eps,
                  BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    md = offs_d < D
    mk = offs_k < K
    x = tl.load(x_ptr + row * K + offs_k, mask=mk, other=0.0)
    w = tl.load(w_ptr + offs_d[:, None] * K + offs_k[None, :],
                mask=md[:, None] & mk[None, :], other=0.0)
    proj = tl.sum(w * x[None, :], axis=1)  # [D]
    r = tl.load(res_ptr + row * D + offs_d, mask=md, other=0.0)
    xv = proj + r
    mean = tl.sum(tl.where(md, xv, 0.0), axis=0) / D
    xc = tl.where(md, xv - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + offs_d, mask=md, other=0.0)
    bb = tl.load(b_ptr + offs_d, mask=md, other=0.0)
    y = xc * rstd * g + bb
    tl.store(out_ptr + row * D + offs_d, y, mask=md)


class MultiHeadAttentionNew(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        d_model = q.size(-1)
        if len_q == len_k == len_v:
            N = sz_b * len_q
            qc = q.reshape(N, d_model)
            kc = k.reshape(N, d_model)
            vc = v.reshape(N, d_model)
            OK = n_head * d_k
            OV = n_head * d_v
            qp = torch.empty((N, OK), device=q.device, dtype=q.dtype)
            kp = torch.empty((N, OK), device=q.device, dtype=q.dtype)
            vp = torch.empty((N, OV), device=q.device, dtype=q.dtype)
            _qkv_kernel[(N,)](
                qc, kc, vc, self.w_qs.weight, self.w_ks.weight, self.w_vs.weight,
                qp, kp, vp, d_model, OK, OV,
                BLOCK_DM=_next_pow2(d_model), BLOCK_OK=_next_pow2(OK),
                BLOCK_OV=_next_pow2(OV), num_warps=4,
            )
            qp = qp.view(sz_b, len_q, n_head, d_k)
            kp = kp.view(sz_b, len_k, n_head, d_k)
            vp = vp.view(sz_b, len_v, n_head, d_v)
        else:
            qp = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            kp = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            vp = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        attn = torch.empty((sz_b, n_head, len_q, len_k), device=q.device, dtype=qp.dtype)

        has_mask = mask is not None
        mask_arg = mask.contiguous() if has_mask else qp
        temperature = float(d_k ** 0.5)

        res = residual.reshape(sz_b * len_q, -1)
        D = res.size(-1)
        y = torch.empty_like(res)

        _fused_attn_fc_ln[(sz_b * len_q,)](
            qp, kp, vp, mask_arg, self.fc.weight, res,
            self.layer_norm.weight, self.layer_norm.bias, y, attn,
            temperature, D, float(self.layer_norm.eps),
            n_head, len_q, len_k, d_k, d_v,
            HAS_MASK=has_mask,
            BLOCK_LK=_next_pow2(len_k), BLOCK_DK=_next_pow2(d_k),
            BLOCK_DV=_next_pow2(d_v), BLOCK_D=_next_pow2(D), num_warps=4,
        )
        y = y.view(sz_b, len_q, D)
        return y, attn
