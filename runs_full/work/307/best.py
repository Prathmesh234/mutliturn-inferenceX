import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _full_kernel(qin_ptr, kin_ptr, vin_ptr,
                 qw_ptr, kw_ptr, vw_ptr,
                 qb_ptr, kb_ptr, vb_ptr,
                 ow_ptr, ob_ptr, out_ptr,
                 M, K, J, hd, size, scale,
                 NUM_HEADS: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                 BLOCK_D: tl.constexpr, BLOCK_O: tl.constexpr):
    b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    offs_o = tl.arange(0, BLOCK_O)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask_d = offs_d < hd
    mask_o = offs_o < size

    in_off = (b * M + offs_m[:, None]) * K + offs_k[None, :]
    in_mask = mask_m[:, None] & mask_k[None, :]
    qin = tl.load(qin_ptr + in_off, mask=in_mask, other=0.0)
    kin = tl.load(kin_ptr + in_off, mask=in_mask, other=0.0)
    vin = tl.load(vin_ptr + in_off, mask=in_mask, other=0.0)

    out_acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for h in range(NUM_HEADS):
        w_off = (h * hd + offs_d[:, None]) * K + offs_k[None, :]
        w_mask = mask_d[:, None] & mask_k[None, :]
        qw = tl.load(qw_ptr + w_off, mask=w_mask, other=0.0)
        kw = tl.load(kw_ptr + w_off, mask=w_mask, other=0.0)
        vw = tl.load(vw_ptr + w_off, mask=w_mask, other=0.0)
        bias_off = h * hd + offs_d
        qb = tl.load(qb_ptr + bias_off, mask=mask_d, other=0.0)
        kb = tl.load(kb_ptr + bias_off, mask=mask_d, other=0.0)
        vb = tl.load(vb_ptr + bias_off, mask=mask_d, other=0.0)

        q_lin = tl.sum(qin[:, None, :] * qw[None, :, :], axis=2) + qb[None, :]
        k_lin = tl.sum(kin[:, None, :] * kw[None, :, :], axis=2) + kb[None, :]
        v_lin = tl.sum(vin[:, None, :] * vw[None, :, :], axis=2) + vb[None, :]

        scores = tl.sum(q_lin[:, None, :] * k_lin[None, :, :], axis=2) * scale
        scores = tl.where(mask_m[None, :], scores, float('-inf'))
        mx = tl.max(scores, axis=1)
        e = tl.exp(scores - mx[:, None])
        s = tl.sum(e, axis=1)
        attn = e / s[:, None]

        ctx = tl.sum(attn[:, :, None] * v_lin[None, :, :], axis=1)  # [M, hd]

        ow_off = offs_o[:, None] * J + h * hd + offs_d[None, :]
        ow = tl.load(ow_ptr + ow_off, mask=mask_o[:, None] & mask_d[None, :], other=0.0)  # [size, hd]
        out_acc += tl.sum(ctx[:, None, :] * ow[None, :, :], axis=2)  # [M, size]

    ob = tl.load(ob_ptr + offs_o, mask=mask_o, other=0.0)
    out_acc += ob[None, :]
    o_off = (b * M + offs_m[:, None]) * size + offs_o[None, :]
    tl.store(out_ptr + o_off, out_acc, mask=mask_m[:, None] & mask_o[None, :])


class MultiHeadedAttentionNew(nn.Module):
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttentionNew, self).__init__()
        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask=None):
        batch_size = k.size(0)
        num_heads = self.num_heads
        hd = self.head_size
        size = self.model_size
        K = size
        J = num_heads * hd
        N = k.numel() // size
        M = N // batch_size

        qf = q.reshape(N, K).contiguous()
        kf = k.reshape(N, K).contiguous()
        vf = v.reshape(N, K).contiguous()

        out = torch.empty((N, size), device=k.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(hd)
        _full_kernel[(batch_size,)](
            qf, kf, vf,
            self.q_layer.weight, self.k_layer.weight, self.v_layer.weight,
            self.q_layer.bias, self.k_layer.bias, self.v_layer.bias,
            self.output_layer.weight, self.output_layer.bias, out,
            M, K, J, hd, size, scale,
            NUM_HEADS=num_heads,
            BLOCK_M=triton.next_power_of_2(M),
            BLOCK_K=triton.next_power_of_2(K),
            BLOCK_D=triton.next_power_of_2(hd),
            BLOCK_O=triton.next_power_of_2(size),
            num_warps=2)
        return out.reshape(batch_size, M, size).to(k.dtype)
