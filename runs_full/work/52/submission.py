import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(hidden_ptr, w1_ptr, w2_ptr, A_ptr, M_ptr,
                  batch, seq, heads, attention_size,
                  HIDDEN_SIZE: tl.constexpr, HID_BLOCK: tl.constexpr,
                  SEQ_BLOCK: tl.constexpr, ATT_BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs_s = tl.arange(0, SEQ_BLOCK)
    offs_a = tl.arange(0, ATT_BLOCK)
    offs_h = tl.arange(0, HID_BLOCK)
    mask_s = offs_s < seq
    mask_a = offs_a < attention_size
    mask_h = offs_h < HIDDEN_SIZE

    # load hidden block [SEQ_BLOCK, HID_BLOCK]
    hb_off = b * seq * HIDDEN_SIZE + offs_s[:, None] * HIDDEN_SIZE + offs_h[None, :]
    hb = tl.load(hidden_ptr + hb_off, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    # x = tanh(hidden @ W1^T) -> [SEQ_BLOCK, ATT_BLOCK]
    acc = tl.zeros((SEQ_BLOCK, ATT_BLOCK), dtype=tl.float32)
    for h in range(HIDDEN_SIZE):
        hv = tl.load(hidden_ptr + b * seq * HIDDEN_SIZE + offs_s * HIDDEN_SIZE + h,
                     mask=mask_s, other=0.0)
        w = tl.load(w1_ptr + offs_a * HIDDEN_SIZE + h, mask=mask_a, other=0.0)
        acc += hv[:, None] * w[None, :]
    x_blk = (tl.exp(2.0 * acc) - 1.0) / (tl.exp(2.0 * acc) + 1.0)

    for head in range(heads):
        w2 = tl.load(w2_ptr + head * attention_size + offs_a, mask=mask_a, other=0.0)
        scores = tl.sum(x_blk * w2[None, :], axis=1)
        scores = tl.where(mask_s, scores, -float('inf'))
        m = tl.max(scores, axis=0)
        e = tl.exp(scores - m)
        e = tl.where(mask_s, e, 0.0)
        a = e / tl.sum(e, axis=0)
        tl.store(A_ptr + (b * heads + head) * seq + offs_s, a, mask=mask_s)
        mvec = tl.sum(a[:, None] * hb, axis=0)  # [HID_BLOCK]
        tl.store(M_ptr + (b * heads + head) * HIDDEN_SIZE + offs_h, mvec, mask=mask_h)


class SelfAttentionNew(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)

    def forward(self, hidden):
        hidden = hidden.transpose(0, 1).contiguous()
        batch, seq, hsz = hidden.shape
        att = self.attention_size
        heads = self.n_attention_heads

        A = torch.empty((batch, heads, seq), device=hidden.device, dtype=hidden.dtype)
        M = torch.empty((batch, heads, hsz), device=hidden.device, dtype=hidden.dtype)

        _fused_kernel[(batch,)](
            hidden, self.W1.weight, self.W2.weight, A, M,
            batch, seq, heads, att,
            HIDDEN_SIZE=hsz, HID_BLOCK=triton.next_power_of_2(hsz),
            SEQ_BLOCK=triton.next_power_of_2(seq),
            ATT_BLOCK=triton.next_power_of_2(att), num_warps=1, num_stages=1)

        return M, A
