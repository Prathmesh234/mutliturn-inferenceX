import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _triplet_kernel(a_ptr, p_ptr, n_ptr, out_ptr, R, K, M, margin, inv,
                    R_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr):
    r = tl.arange(0, R_BLOCK)
    k = tl.arange(0, K_BLOCK)
    rmask = r < R
    kmask = k < K
    n0 = r // M
    m = r % M
    idx = (n0 * K * M + m)[None, :] + (k * M)[:, None]
    mask = kmask[:, None] & rmask[None, :]
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    p = tl.load(p_ptr + idx, mask=mask, other=0.0)
    nn_ = tl.load(n_ptr + idx, mask=mask, other=0.0)
    # (a-p)^2 - (a-n)^2 = (n-p)*(2a - p - n)
    diff = (nn_ - p) * (a + a - p - nn_)
    d = tl.sum(diff, axis=0)
    loss = d + margin
    loss = tl.where((loss > 0) & rmask, loss, 0.0)
    total = tl.sum(loss) * inv
    tl.store(out_ptr, total)


class TripletLossNew(nn.Module):
    def __init__(self, margin):
        super(TripletLossNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        N0 = anchor.shape[0]
        K = anchor.shape[1]
        M = anchor.numel() // (N0 * K)
        R = N0 * M
        R_BLOCK = triton.next_power_of_2(R)
        K_BLOCK = triton.next_power_of_2(K)
        inv = 1.0 / R if size_average else 1.0
        out = torch.empty(1, device=anchor.device, dtype=anchor.dtype)
        _triplet_kernel[(1,)](anchor, positive, negative, out, R, K, M,
                              float(self.margin), inv,
                              R_BLOCK=R_BLOCK, K_BLOCK=K_BLOCK, num_warps=2)
        return out[0]
