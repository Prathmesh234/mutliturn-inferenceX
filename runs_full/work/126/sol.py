import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(x_ptr, out_ptr, S, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    x = tl.load(x_ptr + pid * S + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s / S)


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   sx_m, sx_k, sw_n, sw_k, K, N,
                   HAS_BIAS: tl.constexpr,
                   BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N
    x = tl.load(x_ptr + m * sx_m + offs_k * sx_k, mask=mask_k, other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * sw_n + offs_k[None, :] * sw_k,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)
    acc = tl.sum(x[None, :] * w, axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    tl.store(out_ptr + m * N + offs_n, acc, mask=mask_n)


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                 sqb, sqm, sqc, skb, skn, skc, svb, svn, svc, sob, som, soc,
                 M, N, C, scale,
                 BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    offs_c = tl.arange(0, BLOCK_C)
    offs_n = tl.arange(0, BLOCK_N)
    mask_c = offs_c < C
    mask_n = offs_n < N
    q = tl.load(q_ptr + b * sqb + m * sqm + offs_c * sqc, mask=mask_c, other=0.0)
    k = tl.load(k_ptr + b * skb + offs_n[:, None] * skn + offs_c[None, :] * skc,
                mask=mask_n[:, None] & mask_c[None, :], other=0.0)
    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(mask_n, scores, float('-inf'))
    mx = tl.max(scores, axis=0)
    p = tl.exp(scores - mx)
    p = p / tl.sum(p, axis=0)
    v = tl.load(v_ptr + b * svb + offs_n[:, None] * svn + offs_c[None, :] * svc,
                mask=mask_n[:, None] & mask_c[None, :], other=0.0)
    out = tl.sum(p[:, None] * v, axis=0)
    tl.store(out_ptr + b * sob + m * som + offs_c * soc, out, mask=mask_c)


@triton.jit
def _badd_kernel(a_ptr, b_ptr, out_ptr, N, H, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    bb = pid // N
    nn = pid % N
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    a = tl.load(a_ptr + nn * H + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + bb * H + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid * H + offs, a + b, mask=mask)


@triton.jit
def _film_kernel(x_ptr, film_ptr, gs_ptr, gb_ptr, bs_ptr, out_ptr,
                 total, C, HW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    c_idx = (offs // HW) % C
    b_idx = offs // (C * HW)
    xval = tl.load(x_ptr + offs, mask=mask, other=0.0)
    g = tl.load(film_ptr + b_idx * (2 * C) + c_idx, mask=mask, other=0.0)
    be = tl.load(film_ptr + b_idx * (2 * C) + C + c_idx, mask=mask, other=0.0)
    gs = tl.load(gs_ptr + c_idx, mask=mask, other=0.0)
    gb = tl.load(gb_ptr + c_idx, mask=mask, other=0.0)
    bs = tl.load(bs_ptr + c_idx, mask=mask, other=0.0)
    gamma = g * gs + gb
    beta = be * bs
    tl.store(out_ptr + offs, gamma * xval + beta, mask=mask)


def _linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _linear_kernel[(M,)](
        x, weight, bias if bias is not None else weight, out,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), K, N,
        bias is not None,
        BLOCK_K=triton.next_power_of_2(K), BLOCK_N=triton.next_power_of_2(N),
        num_warps=4)
    return out


def _attention(q, k, v, scale):
    Bq, M, C = q.shape
    Bk, N, _ = k.shape
    skb = k.stride(0) if Bk > 1 else 0
    svb = v.stride(0) if Bk > 1 else 0
    out = torch.empty((Bq, M, C), device=q.device, dtype=q.dtype)
    _attn_kernel[(Bq * M,)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        skb, k.stride(1), k.stride(2),
        svb, v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        M, N, C, scale,
        BLOCK_N=triton.next_power_of_2(N), BLOCK_C=triton.next_power_of_2(C),
        num_warps=4)
    return out


class AttentionModuleV2New(torch.nn.Module):

    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None,
        fc_spt_value=None, fc_x_update=None, fc_update=None,
        fc_spt_spt_query=None, fc_spt_spt_key=None, fc_spt_spt_value=None,
        gamma_scale_gate=None, gamma_bias_gate=None, beta_scale_gate=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_x_query = fc_x_query if fc_x_query is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_key = fc_spt_key if fc_spt_key is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_value = fc_spt_value if fc_spt_value is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_x_update = fc_x_update if fc_x_update is not None else torch.nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.fc_update = fc_update if fc_update is not None else torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.fc_spt_spt_query = fc_spt_spt_query if fc_spt_spt_query is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_key = fc_spt_spt_key if fc_spt_spt_key is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_value = fc_spt_spt_value if fc_spt_spt_value is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma_scale_gate = gamma_scale_gate if gamma_scale_gate is not None else torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))
        self.gamma_bias_gate = gamma_bias_gate if gamma_bias_gate is not None else torch.nn.Parameter(torch.ones(size=[1, hidden_size, 1, 1, 1], requires_grad=True))
        self.beta_scale_gate = beta_scale_gate if beta_scale_gate is not None else torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

    def forward(self, x, proto_spt):
        B, C, Hs, Ws = x.shape
        H = self.hidden_size
        N = proto_spt.shape[0]
        scale = 1.0 / math.sqrt(H)

        # proto_x = mean over spatial -> [B, H]
        xc = x.contiguous()
        proto_x = torch.empty((B, C), device=x.device, dtype=x.dtype)
        _mean_kernel[(B * C,)](xc, proto_x, Hs * Ws,
                               BLOCK_S=triton.next_power_of_2(Hs * Ws))
        spt = proto_spt.contiguous()  # [N, H]

        # Attention 1
        q = _linear(proto_x, self.fc_x_query.weight, None)          # [B,H]
        k = _linear(spt, self.fc_spt_key.weight, None)              # [N,H]
        v = _linear(spt, self.fc_spt_value.weight, None)            # [N,H]
        agg = _attention(q.view(B, 1, H), k.view(1, N, H), v.view(1, N, H), scale).view(B, H)
        proto_x = _linear(torch.cat([proto_x, agg], dim=-1),
                          self.fc_x_update.weight, self.fc_x_update.bias)  # [B,H]

        # proto_spt = spt[N,H] + proto_x[B,H] -> [B,N,H]
        ps = torch.empty((B, N, H), device=x.device, dtype=x.dtype)
        _badd_kernel[(B * N,)](spt, proto_x, ps, N, H,
                               BLOCK_H=triton.next_power_of_2(H))
        psf = ps.view(B * N, H)

        # Attention 2 (self)
        q2 = _linear(psf, self.fc_spt_spt_query.weight, None).view(B, N, H)
        k2 = _linear(psf, self.fc_spt_spt_key.weight, None).view(B, N, H)
        v2 = _linear(psf, self.fc_spt_spt_value.weight, None).view(B, N, H)
        proto_spt2 = _attention(q2, k2, v2, scale)  # [B,N,H]
        ps2f = proto_spt2.reshape(B * N, H)

        # Attention 3
        q3 = _linear(proto_x, self.fc_x_query.weight, None)            # [B,H]
        k3 = _linear(ps2f, self.fc_spt_key.weight, None).view(B, N, H)
        v3 = _linear(ps2f, self.fc_spt_value.weight, None).view(B, N, H)
        agg3 = _attention(q3.view(B, 1, H), k3, v3, scale).view(B, H)

        film = _linear(torch.cat([proto_x, agg3], dim=-1),
                       self.fc_update.weight, self.fc_update.bias)  # [B,2H]
        film = film.contiguous()

        gs = self.gamma_scale_gate.reshape(-1).contiguous()
        gb = self.gamma_bias_gate.reshape(-1).contiguous()
        bs = self.beta_scale_gate.reshape(-1).contiguous()

        out = torch.empty_like(xc)
        total = B * C * Hs * Ws
        _film_kernel[(triton.cdiv(total, 1024),)](
            xc, film, gs, gb, bs, out, total, C, Hs * Ws, BLOCK=1024,
            num_warps=4)
        return out
