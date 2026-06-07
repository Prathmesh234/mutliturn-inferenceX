import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(A, B, Bias, C, M, N, K,
                 sam, sak, sbk, sbn, scm, scn,
                 HAS_BIAS: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kk = k * BK + offk
        a = tl.load(A + offm[:, None] * sam + kk[None, :] * sak,
                    mask=(offm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * sbk + offn[None, :] * sbn,
                    mask=(kk[:, None] < K) & (offn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    if HAS_BIAS:
        acc += tl.load(Bias + offn, mask=offn < N, other=0.0)[None, :]
    tl.store(C + offm[:, None] * scm + offn[None, :] * scn, acc,
             mask=(offm[:, None] < M) & (offn[None, :] < N))


@triton.jit
def _s_kernel(C, E, S, seq, depth, heads, n_state, max_seq,
              sc_b, sc_s, se_h, se_m, se_d, ss_b, ss_h, ss_r, ss_m,
              BS: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // heads
    h = pid % heads
    offs_s = tl.arange(0, BS)
    offs_d = tl.arange(0, BD)
    ms = offs_s < seq
    md = offs_d < depth
    q_ptr = C + b * sc_b + offs_s[:, None] * sc_s + (h * depth + offs_d[None, :])
    Q = tl.load(q_ptr, mask=ms[:, None] & md[None, :], other=0.0)
    e_ptr = E + h * se_h + (max_seq - seq + offs_s[:, None]) * se_m + offs_d[None, :] * se_d
    Erow = tl.load(e_ptr, mask=ms[:, None] & md[None, :], other=0.0)
    Sm = tl.dot(Q, tl.trans(Erow), input_precision="ieee")
    s_ptr = S + b * ss_b + h * ss_h + offs_s[:, None] * ss_r + offs_s[None, :] * ss_m
    tl.store(s_ptr, Sm, mask=ms[:, None] & ms[None, :])


@triton.jit
def _attn_kernel(C, S, Out, Mask, seq, depth, heads, n_state, scale,
                 sc_b, sc_s, ss_b, ss_h, ss_r, ss_m, so_b, so_s,
                 sm_b, sm_h, sm_i, sm_j,
                 HAS_MASK: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // heads
    h = pid % heads
    offs_s = tl.arange(0, BS)
    offs_d = tl.arange(0, BD)
    ms = offs_s < seq
    md = offs_d < depth

    base = C + b * sc_b
    Q = tl.load(base + offs_s[:, None] * sc_s + (h * depth + offs_d[None, :]),
                mask=ms[:, None] & md[None, :], other=0.0)
    K = tl.load(base + offs_s[:, None] * sc_s + (n_state + h * depth + offs_d[None, :]),
                mask=ms[:, None] & md[None, :], other=0.0)
    V = tl.load(base + offs_s[:, None] * sc_s + (2 * n_state + h * depth + offs_d[None, :]),
                mask=ms[:, None] & md[None, :], other=0.0)

    W = tl.dot(Q, tl.trans(K), input_precision="ieee")

    i = offs_s[:, None]
    j = offs_s[None, :]
    p = (i + 1) * seq + j
    mod = p % (seq + 1)
    r = p // (seq + 1)
    col = mod - 1
    valid = (mod != 0) & ms[:, None] & ms[None, :]
    rel = tl.load(S + b * ss_b + h * ss_h + r * ss_r + col * ss_m,
                  mask=valid, other=0.0)

    w = (W + rel) * scale
    if HAS_MASK:
        mk = tl.load(Mask + b * sm_b + h * sm_h + i * sm_i + j * sm_j,
                     mask=ms[:, None] & ms[None, :], other=0.0)
        w = w + mk

    w = tl.where(ms[None, :], w, -1e30)
    mx = tl.max(w, axis=1)[:, None]
    e = tl.exp(w - mx)
    denom = tl.sum(e, axis=1)[:, None]
    P = e / denom

    a = tl.dot(P, V, input_precision="ieee")
    o_ptr = Out + b * so_b + offs_s[:, None] * so_s + (h * depth + offs_d[None, :])
    tl.store(o_ptr, a, mask=ms[:, None] & md[None, :])


def _gemm(A, B, bias, M, N, K):
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_kernel[grid](A, B, bias if bias is not None else A, C, M, N, K,
                       A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                       C.stride(0), C.stride(1),
                       HAS_BIAS=bias is not None,
                       BM=BM, BN=BN, BK=BK, num_warps=2)
    return C


class RelativeAttentionNew(nn.Module):
    def __init__(self, heads, n_state, max_sequence):
        super().__init__()
        assert n_state % heads == 0
        self.heads = heads
        self.n_state = n_state
        self.depth = n_state // heads
        self.max_sequence = max_sequence

        class _Conv1d(nn.Module):
            def __init__(s, nf, nx, stdev=0.02):
                super().__init__()
                s.nf = nf
                s.nx = nx
                s.stdev = stdev
                s.w = nn.Parameter(torch.normal(size=[1, nx, nf], mean=0.0, std=stdev))
                s.b = nn.Parameter(torch.zeros([nf]))

        self.c_attn = _Conv1d(n_state * 3, n_state)
        self.c_proj = _Conv1d(n_state, n_state)
        self.E = nn.Parameter(torch.Tensor(self.heads, self.max_sequence, n_state // heads))
        nn.init.xavier_normal_(self.E)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        n_state = self.n_state
        heads = self.heads
        depth = self.depth

        x2 = x.reshape(-1, n_state).contiguous()
        wa = self.c_attn.w.reshape(n_state, 3 * n_state)
        c2 = _gemm(x2, wa, self.c_attn.b, batch * seq_len, 3 * n_state, n_state)
        c = c2.reshape(batch, seq_len, 3 * n_state)

        S = torch.empty((batch, heads, seq_len, seq_len), device=x.device, dtype=torch.float32)
        BS = max(16, triton.next_power_of_2(seq_len))
        BD = max(16, triton.next_power_of_2(depth))
        grid = (batch * heads,)
        E = self.E
        _s_kernel[grid](c, E, S, seq_len, depth, heads, n_state, self.max_sequence,
                        c.stride(0), c.stride(1),
                        E.stride(0), E.stride(1), E.stride(2),
                        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
                        BS=BS, BD=BD, num_warps=1)

        out = torch.empty((batch, seq_len, n_state), device=x.device, dtype=torch.float32)
        scale = 1.0 / (depth ** 0.5)

        if mask is not None:
            mask_b = torch.broadcast_to(mask, (batch, heads, seq_len, seq_len)).contiguous()
            sm = (mask_b.stride(0), mask_b.stride(1), mask_b.stride(2), mask_b.stride(3))
            has_mask = True
        else:
            mask_b = c
            sm = (0, 0, 0, 0)
            has_mask = False

        _attn_kernel[grid](c, S, out, mask_b, seq_len, depth, heads, n_state, scale,
                           c.stride(0), c.stride(1),
                           S.stride(0), S.stride(1), S.stride(2), S.stride(3),
                           out.stride(0), out.stride(1),
                           sm[0], sm[1], sm[2], sm[3],
                           HAS_MASK=has_mask, BS=BS, BD=BD, num_warps=1)

        out2 = out.reshape(-1, n_state)
        wp = self.c_proj.w.reshape(n_state, n_state)
        r2 = _gemm(out2, wp, self.c_proj.b, batch * seq_len, n_state, n_state)
        return r2.reshape(batch, seq_len, n_state)
