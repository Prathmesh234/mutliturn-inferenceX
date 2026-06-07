import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(X, H_, W, Wb, U, Ub, OUT,
                  M, KX, N,
                  sxm, sxk, shm, shk, swk, swn, suk, sun, som, son,
                  BM: tl.constexpr, BN: tl.constexpr, BKX: tl.constexpr, BKH: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    # p = X @ W + Wb
    offs_kx = tl.arange(0, BKX)
    x_ptrs = X + offs_m[:, None] * sxm + offs_kx[None, :] * sxk
    w_ptrs = W + offs_kx[:, None] * swk + offs_n[None, :] * swn
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_kx[None, :] < KX), other=0.0)
    w = tl.load(w_ptrs, mask=(offs_kx[:, None] < KX) & (offs_n[None, :] < N), other=0.0)
    p = tl.dot(x, w)
    wb = tl.load(Wb + offs_n, mask=offs_n < N, other=0.0)
    p += wb[None, :]
    # q = H_ @ U + Ub  (K = N = hidden)
    offs_kh = tl.arange(0, BKH)
    h_ptrs = H_ + offs_m[:, None] * shm + offs_kh[None, :] * shk
    u_ptrs = U + offs_kh[:, None] * suk + offs_n[None, :] * sun
    hm = tl.load(h_ptrs, mask=(offs_m[:, None] < M) & (offs_kh[None, :] < N), other=0.0)
    u = tl.load(u_ptrs, mask=(offs_kh[:, None] < N) & (offs_n[None, :] < N), other=0.0)
    q = tl.dot(hm, u)
    ub = tl.load(Ub + offs_n, mask=offs_n < N, other=0.0)
    q += ub[None, :]
    # h_ values (for elementwise) -- column index must be < N (hidden)
    h_e = tl.load(H_ + offs_m[:, None] * shm + offs_n[None, :] * shk,
                  mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    i = tl.sigmoid(p + q)
    f = tl.sigmoid(p - q)
    z = i * p + f * h_e
    h = 2.0 * tl.sigmoid(2.0 * z) - 1.0
    o_ptrs = OUT + offs_m[:, None] * som + offs_n[None, :] * son
    tl.store(o_ptrs, h, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _np2(x):
    return max(16, triton.next_power_of_2(x))


class ATRCellNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ATRCellNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):
        x = x.contiguous()
        h_ = h_.contiguous()
        M, KX = x.shape
        N = self.hidden_size
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BN = _np2(N)
        BKX = _np2(KX)
        BKH = _np2(N)
        BM = 16 if M <= 16 else 64
        grid = (triton.cdiv(M, BM),)
        _fused_kernel[grid](x, h_, self._W, self._W_b, self._U, self._U_b, out,
                            M, KX, N,
                            x.stride(0), x.stride(1), h_.stride(0), h_.stride(1),
                            self._W.stride(0), self._W.stride(1),
                            self._U.stride(0), self._U.stride(1),
                            out.stride(0), out.stride(1),
                            BM=BM, BN=BN, BKX=BKX, BKH=BKH, num_warps=1)
        return out
