import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _lrn_kernel(x_ptr, w_ptr, b_ptr, h_ptr, out_ptr,
                B, I, H,
                stride_xb, stride_xi,
                stride_wi, stride_wn,
                stride_hb, stride_hh,
                stride_ob, stride_oh,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, I, BLOCK_K):
        kk = k + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * stride_xb + kk[None, :] * stride_xi,
                    mask=(offs_m[:, None] < B) & (kk[None, :] < I), other=0.0)
        wp = tl.load(w_ptr + kk[:, None] * stride_wi + offs_n[None, :] * stride_wn,
                     mask=(kk[:, None] < I) & (offs_n[None, :] < H), other=0.0)
        wq = tl.load(w_ptr + kk[:, None] * stride_wi + (offs_n[None, :] + H) * stride_wn,
                     mask=(kk[:, None] < I) & (offs_n[None, :] < H), other=0.0)
        wr = tl.load(w_ptr + kk[:, None] * stride_wi + (offs_n[None, :] + 2 * H) * stride_wn,
                     mask=(kk[:, None] < I) & (offs_n[None, :] < H), other=0.0)
        accp += tl.dot(a, wp)
        accq += tl.dot(a, wq)
        accr += tl.dot(a, wr)

    bp = tl.load(b_ptr + offs_n, mask=offs_n < H, other=0.0)
    bq = tl.load(b_ptr + offs_n + H, mask=offs_n < H, other=0.0)
    br = tl.load(b_ptr + offs_n + 2 * H, mask=offs_n < H, other=0.0)
    accp += bp[None, :]
    accq += bq[None, :]
    accr += br[None, :]

    hmask = (offs_m[:, None] < B) & (offs_n[None, :] < H)
    h_ = tl.load(h_ptr + offs_m[:, None] * stride_hb + offs_n[None, :] * stride_hh,
                 mask=hmask, other=0.0)

    i = tl.sigmoid(accp + h_)
    f = tl.sigmoid(accq - h_)
    out = _tanh(i * accr + f * h_)
    tl.store(out_ptr + offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oh,
             out, mask=hmask)


class LRNCellNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LRNCellNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size * 3))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, h_):
        x = x.contiguous()
        h_ = h_.contiguous()
        B = x.shape[0]
        I = self.input_size
        H = self.hidden_size
        out = torch.empty((B, H), device=x.device, dtype=x.dtype)
        BLOCK_M = 16
        BLOCK_N = 16
        BLOCK_K = 16
        grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _lrn_kernel[grid](x, self._W, self._W_b, h_, out,
                          B, I, H,
                          x.stride(0), x.stride(1),
                          self._W.stride(0), self._W.stride(1),
                          h_.stride(0), h_.stride(1),
                          out.stride(0), out.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                          num_warps=1)
        return out
