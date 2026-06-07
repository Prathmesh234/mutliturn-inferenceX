import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _score_kernel(hidden_ptr, enc_ptr, W_ptr, bias_ptr, v_ptr, out_ptr,
                  B, T, H: tl.constexpr, H2: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T
    offs = tl.arange(0, BH)
    m = offs < H
    hb = tl.load(hidden_ptr + b * H + offs, mask=m, other=0.0)
    enc = tl.load(enc_ptr + t * (B * H) + b * H + offs, mask=m, other=0.0)
    bias = tl.load(bias_ptr + offs, mask=m, other=0.0)
    v = tl.load(v_ptr + offs, mask=m, other=0.0)
    ho = offs[:, None]
    k = offs[None, :]
    wm = (ho < H) & (k < H)
    w1 = tl.load(W_ptr + ho * H2 + k, mask=wm, other=0.0)
    w2 = tl.load(W_ptr + ho * H2 + (H + k), mask=wm, other=0.0)
    pre = bias + tl.sum(w1 * hb[None, :], axis=1) + tl.sum(w2 * enc[None, :], axis=1)
    e = (1.0 - tl.exp(-2.0 * pre)) / (1.0 + tl.exp(-2.0 * pre))
    tl.store(out_ptr + b * T + t, tl.sum(v * e, axis=0))


@triton.jit
def _softmax_kernel(in_ptr, out_ptr, B, T, BT: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BT)
    m = offs < T
    x = tl.load(in_ptr + b * T + offs, mask=m, other=-float('inf'))
    mx = tl.max(x, axis=0)
    e = tl.where(m, tl.exp(x - mx), 0.0)
    tl.store(out_ptr + b * T + offs, e / tl.sum(e, axis=0), mask=m)


@triton.jit
def _fused_kernel(hidden_ptr, enc_ptr, W_ptr, bias_ptr, v_ptr, out_ptr,
                  B, T, H: tl.constexpr, H2: tl.constexpr,
                  BH: tl.constexpr, BT: tl.constexpr):
    b = tl.program_id(0)
    offs_h = tl.arange(0, BH)
    offs_t = tl.arange(0, BT)
    mh = offs_h < H
    mt = offs_t < T

    hb = tl.load(hidden_ptr + b * H + offs_h, mask=mh, other=0.0)
    bias = tl.load(bias_ptr + offs_h, mask=mh, other=0.0)
    v = tl.load(v_ptr + offs_h, mask=mh, other=0.0)

    ho = offs_h[:, None]
    k = offs_h[None, :]
    wm = (ho < H) & (k < H)
    w1 = tl.load(W_ptr + ho * H2 + k, mask=wm, other=0.0)
    w2 = tl.load(W_ptr + ho * H2 + (H + k), mask=wm, other=0.0)

    enc = tl.load(enc_ptr + offs_t[:, None] * (B * H) + b * H + offs_h[None, :],
                  mask=mt[:, None] & mh[None, :], other=0.0)

    hidden_term = tl.sum(w1 * hb[None, :], axis=1)               # [BH]
    enc_term = tl.sum(enc[:, None, :] * w2[None, :, :], axis=2)  # [BT, BH]
    pre = bias[None, :] + hidden_term[None, :] + enc_term        # [BT, BH]
    e = (1.0 - tl.exp(-2.0 * pre)) / (1.0 + tl.exp(-2.0 * pre))
    score = tl.sum(v[None, :] * e, axis=1)                       # [BT]

    score = tl.where(mt, score, -float('inf'))
    mx = tl.max(score, axis=0)
    ex = tl.where(mt, tl.exp(score - mx), 0.0)
    tl.store(out_ptr + b * T + offs_t, ex / tl.sum(ex, axis=0), mask=mt)


class BahdanauAttentionNew(nn.Module):

    def __init__(self, hidden_size):
        super(BahdanauAttentionNew, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, mask=None):
        T = encoder_outputs.size(0)
        B = hidden.size(0)
        H = self.hidden_size
        out = torch.empty((B, 1, T), device=hidden.device, dtype=hidden.dtype)
        BH = triton.next_power_of_2(H)
        BT = triton.next_power_of_2(T)
        if mask is None:
            _fused_kernel[(B,)](hidden, encoder_outputs, self.attn.weight,
                                self.attn.bias, self.v,
                                out.view(B, T), B, T, H, 2 * H,
                                BH=BH, BT=BT, num_warps=1)
        else:
            scores = torch.empty((B, T), device=hidden.device, dtype=hidden.dtype)
            _score_kernel[(B * T,)](hidden, encoder_outputs, self.attn.weight,
                                    self.attn.bias, self.v, scores,
                                    B, T, H, 2 * H, BH=BH, num_warps=4)
            scores.masked_fill_(mask, -float('inf'))
            _softmax_kernel[(B,)](scores, out.view(B, T), B, T, BT=BT, num_warps=4)
        return out
