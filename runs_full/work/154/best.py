import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_delta(x_ptr, w_ptr, out_ptr, M, L, OW,
                 ORDERP1: tl.constexpr, BLOCK_M: tl.constexpr,
                 L_POW2: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, L_POW2)
    rmask = rows < M
    cmask = cols < L
    rc = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + rows[:, None] * L + cols[None, :], mask=rc, other=0.0)
    wmask = (cols[:, None] < L) & (cols[None, :] < L)
    for j in tl.static_range(ORDERP1):
        w = tl.load(w_ptr + j * L * L + cols[:, None] * L + cols[None, :],
                    mask=wmask, other=0.0)
        out = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        tl.store(out_ptr + rows[:, None] * OW + (j * L + cols)[None, :],
                 out, mask=rc)


class DeltaNew(nn.Module):

    def __init__(self, order=2, **kwargs):
        super(DeltaNew, self).__init__()
        self.order = order
        self.win_length = int(kwargs.get('win_length', 5))
        self._wcache = {}

    def _wpow(self, L, device, dtype):
        key = (L, device, dtype)
        Wp = self._wcache.get(key)
        if Wp is None:
            n = (self.win_length - 1) // 2
            denom = n * (n + 1) * (2 * n + 1) / 3.0
            W = torch.zeros(L, L, dtype=torch.float64)
            for i in range(L):
                for t in range(-n, n + 1):
                    k = min(max(i + t, 0), L - 1)
                    W[i, k] += t / denom
            mats = [torch.eye(L, dtype=torch.float64)]
            for _ in range(self.order):
                mats.append(W @ mats[-1])
            Wp = torch.stack(mats, 0).to(device=device, dtype=dtype).contiguous()
            self._wcache[key] = Wp
        return Wp

    def forward(self, x):
        orig = x.shape
        L = orig[-1]
        M = x.numel() // L
        OW = (self.order + 1) * L
        Wp = self._wpow(L, x.device, x.dtype)
        xf = x.reshape(M, L).contiguous()
        out = torch.empty((M, OW), device=x.device, dtype=x.dtype)
        L_POW2 = triton.next_power_of_2(L)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _fused_delta[grid](xf, Wp, out, M, L, OW,
                           ORDERP1=self.order + 1, BLOCK_M=BLOCK_M,
                           L_POW2=L_POW2, num_warps=1)
        return out.reshape(*orig[:-1], OW)
