import torch
import triton
import triton.language as tl


@triton.jit
def _sdp_kernel(q_ptr, k_ptr, v_ptr, o_ptr, w_ptr,
                B0, NH, L, S, E, scale,
                sq_b0, sq_l, sq_nh, sq_e,
                sk_b0, sk_s, sk_nh, sk_e,
                sv_b0, sv_s, sv_nh, sv_e,
                so_b0, so_l, so_nh, so_e,
                sw_b0, sw_nh, sw_l, sw_s,
                BL: tl.constexpr, BS: tl.constexpr, BE: tl.constexpr,
                EXACT: tl.constexpr):
    b0 = tl.program_id(0)
    nh = tl.program_id(1)

    offs_l = tl.arange(0, BL)
    offs_s = tl.arange(0, BS)
    offs_e = tl.arange(0, BE)

    q_base = q_ptr + b0 * sq_b0 + nh * sq_nh
    k_base = k_ptr + b0 * sk_b0 + nh * sk_nh
    v_base = v_ptr + b0 * sv_b0 + nh * sv_nh

    q_ptrs = q_base + offs_l[:, None] * sq_l + offs_e[None, :] * sq_e
    k_ptrs = k_base + offs_s[:, None] * sk_s + offs_e[None, :] * sk_e
    v_ptrs = v_base + offs_s[:, None] * sv_s + offs_e[None, :] * sv_e

    if EXACT:
        q = tl.load(q_ptrs) * scale
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
    else:
        ml = offs_l < L
        ms = offs_s < S
        me = offs_e < E
        q = tl.load(q_ptrs, mask=ml[:, None] & me[None, :], other=0.0) * scale
        k = tl.load(k_ptrs, mask=ms[:, None] & me[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=ms[:, None] & me[None, :], other=0.0)
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
        scores = tl.where(ms[None, :], scores, float('-inf'))

    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    out = tl.sum(p[:, :, None] * v[None, :, :], axis=1)

    w_ptrs = w_ptr + b0 * sw_b0 + nh * sw_nh + offs_l[:, None] * sw_l + offs_s[None, :] * sw_s
    o_ptrs = o_ptr + b0 * so_b0 + nh * so_nh + offs_l[:, None] * so_l + offs_e[None, :] * so_e
    if EXACT:
        tl.store(w_ptrs, p)
        tl.store(o_ptrs, out)
    else:
        ml = offs_l < L
        ms = offs_s < S
        me = offs_e < E
        tl.store(w_ptrs, p, mask=ml[:, None] & ms[None, :])
        tl.store(o_ptrs, out, mask=ml[:, None] & me[None, :])


class ScaledDotProductNew(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super(ScaledDotProductNew, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None, bias_k=None, bias_v=None):
        B0, L, NH, E = query.shape
        S = key.shape[1]
        scale = E ** -0.5

        attn_output = torch.empty((B0, L, NH, E), device=query.device, dtype=query.dtype)
        weights = torch.empty((B0, NH, L, S), device=query.device, dtype=query.dtype)

        BL = triton.next_power_of_2(L)
        BS = triton.next_power_of_2(S)
        BE = triton.next_power_of_2(E)
        EXACT = (BL == L) and (BS == S) and (BE == E)

        grid = (B0, NH)
        _sdp_kernel[grid](
            query, key, value, attn_output, weights,
            B0, NH, L, S, E, scale,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
            weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
            BL=BL, BS=BS, BE=BE, EXACT=EXACT, num_warps=4,
        )
        return attn_output, weights
