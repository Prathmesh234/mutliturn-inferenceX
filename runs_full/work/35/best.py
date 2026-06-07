import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(qf_ptr, kf_ptr, vf_ptr, wq_ptr, wk_ptr, wv_ptr,
                  a_ptr, r_ptr, fw_ptr, lw_ptr, lb_ptr, o_ptr,
                  bias_ptr, HAS_BIAS: tl.constexpr,
                  n_head, L, K, Dk, Dv, C, F, temperature, eps,
                  sxr, swr, sfwr, sr, so, sab, saq, sak, sbb, sbk,
                  BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
                  BLOCK_K: tl.constexpr, BLOCK_F: tl.constexpr):
    b = tl.program_id(0)
    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    offs_f = tl.arange(0, BLOCK_F)

    xq = tl.load(qf_ptr + (b * L + offs_l)[:, None] * sxr + offs_k[None, :],
                 mask=(offs_l[:, None] < L) & (offs_k[None, :] < K), other=0.0)
    xk = tl.load(kf_ptr + (b * L + offs_l)[:, None] * sxr + offs_k[None, :],
                 mask=(offs_l[:, None] < L) & (offs_k[None, :] < K), other=0.0)
    xv = tl.load(vf_ptr + (b * L + offs_l)[:, None] * sxr + offs_k[None, :],
                 mask=(offs_l[:, None] < L) & (offs_k[None, :] < K), other=0.0)

    fc_acc = tl.zeros((BLOCK_L, BLOCK_F), tl.float32)
    for h in range(n_head):
        wqh = tl.load(wq_ptr + (h * Dk + offs_d)[:, None] * swr + offs_k[None, :],
                      mask=(offs_d[:, None] < Dk) & (offs_k[None, :] < K), other=0.0)
        wkh = tl.load(wk_ptr + (h * Dk + offs_d)[:, None] * swr + offs_k[None, :],
                      mask=(offs_d[:, None] < Dk) & (offs_k[None, :] < K), other=0.0)
        wvh = tl.load(wv_ptr + (h * Dv + offs_d)[:, None] * swr + offs_k[None, :],
                      mask=(offs_d[:, None] < Dv) & (offs_k[None, :] < K), other=0.0)
        qh = tl.dot(xq, tl.trans(wqh))
        kh = tl.dot(xk, tl.trans(wkh))
        vh = tl.dot(xv, tl.trans(wvh))

        scores = tl.dot(qh, tl.trans(kh)) / temperature
        scores = tl.where(offs_l[None, :] < L, scores, -1e9)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + (b * n_head + h) * sbb + offs_l[None, :] * sbk,
                           mask=offs_l[None, :] < L, other=0.0)
            scores = scores + bias
        m = tl.max(scores, axis=1)[:, None]
        p = tl.exp(scores - m)
        s = tl.sum(p, axis=1)[:, None]
        p = p / s
        tl.store(a_ptr + (b * n_head + h) * sab + offs_l[:, None] * saq + offs_l[None, :] * sak, p,
                 mask=(offs_l[:, None] < L) & (offs_l[None, :] < L))
        outh = tl.dot(p, vh)
        wh = tl.load(fw_ptr + offs_f[:, None] * sfwr + (h * Dv + offs_d[None, :]),
                     mask=(offs_f[:, None] < F) & (offs_d[None, :] < Dv), other=0.0)
        fc_acc += tl.dot(outh, tl.trans(wh))

    r = tl.load(r_ptr + (b * L + offs_l)[:, None] * sr + offs_f[None, :],
                mask=(offs_l[:, None] < L) & (offs_f[None, :] < F), other=0.0)
    a = fc_acc + r
    fmask = offs_f[None, :] < F
    mean = tl.sum(tl.where(fmask, a, 0.0), axis=1)[:, None] / F
    ac = tl.where(fmask, a - mean, 0.0)
    var = tl.sum(ac * ac, axis=1)[:, None] / F
    norm = ac / tl.sqrt(var + eps)
    lw = tl.load(lw_ptr + offs_f, mask=offs_f < F, other=0.0)
    lb = tl.load(lb_ptr + offs_f, mask=offs_f < F, other=0.0)
    o = norm * lw[None, :] + lb[None, :]
    tl.store(o_ptr + (b * L + offs_l)[:, None] * so + offs_f[None, :], o,
             mask=(offs_l[:, None] < L) & (offs_f[None, :] < F))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)


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
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        d_model = q.size(2)
        residual = q

        assert len_q == len_k == len_v
        L = len_q
        qf = q.reshape(sz_b * L, d_model).contiguous()
        kf = k.reshape(sz_b * L, d_model).contiguous()
        vf = v.reshape(sz_b * L, d_model).contiguous()

        BH = sz_b * n_head
        BLOCK_L = max(16, triton.next_power_of_2(L))
        BLOCK_D = max(16, triton.next_power_of_2(max(d_k, d_v)))
        BLOCK_K = max(16, triton.next_power_of_2(d_model))
        BLOCK_F = max(16, triton.next_power_of_2(d_model))

        attn_mat = torch.empty((BH, L, L), device=q.device, dtype=torch.float32)
        out = torch.empty((sz_b * L, d_model), device=q.device, dtype=torch.float32)
        res = qf  # residual == qf (same data, contiguous)

        has_bias = mask is not None
        if has_bias:
            mm = mask.unsqueeze(1).to(torch.float32)
            bias = torch.where(mm == 0, torch.full_like(mm, -1e9), torch.zeros_like(mm))
            bias = bias.expand(sz_b, n_head, 1, L).reshape(BH, L).contiguous()
            sbb, sbk = bias.stride(0), bias.stride(1)
        else:
            bias = torch.empty((1,), device=q.device, dtype=torch.float32)
            sbb, sbk = 0, 0

        C = n_head * d_v
        _fused_kernel[(sz_b,)](
            qf, kf, vf, self.w_qs.weight, self.w_ks.weight, self.w_vs.weight,
            attn_mat, res, self.fc.weight, self.layer_norm.weight, self.layer_norm.bias, out,
            bias, has_bias,
            n_head, L, d_model, d_k, d_v, C, d_model, float(d_k ** 0.5), 1e-6,
            qf.stride(0), self.w_qs.weight.stride(0), self.fc.weight.stride(0),
            res.stride(0), out.stride(0),
            attn_mat.stride(0), attn_mat.stride(1), attn_mat.stride(2),
            sbb, sbk,
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, BLOCK_F=BLOCK_F, num_warps=4)

        attn_mat = attn_mat.view(sz_b, n_head, L, L)
        out = out.view(sz_b, L, d_model)
        return out, attn_mat
