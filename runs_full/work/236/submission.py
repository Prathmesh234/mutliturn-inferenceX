import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, t_ptr, out_ptr, p_ptr, w_ptr, N, HW, smooth, eps,
                  BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    d = tl.program_id(0)
    m = tl.program_id(1)
    pb = d * 8
    a1 = tl.load(p_ptr + pb + 0)
    a2 = tl.load(p_ptr + pb + 1)
    st0 = tl.load(p_ptr + pb + 2)
    st1 = tl.load(p_ptr + pb + 3)
    st2 = tl.load(p_ptr + pb + 4)
    sr = tl.load(p_ptr + pb + 5)
    R = tl.load(p_ptr + pb + 6)
    M = tl.load(p_ptr + pb + 7)
    Mf = M.to(tl.float32)
    a12 = a1 * a2
    i0 = m // a12
    rr = m % a12
    i1 = rr // a2
    i2 = rr % a2
    base = i0 * st0 + i1 * st1 + i2 * st2
    offs = tl.arange(0, BLOCK_R)                    # [BR]
    valid_m = m < M
    elem_ok = (offs < R) & valid_m                  # [BR]
    o = base + offs * sr                            # [BR] element offsets
    # inline softmax over N axis for each element
    n = (o // HW) % N                               # [BR]
    base_n = o - n * HW                             # [BR]
    nn = tl.arange(0, BLOCK_N)                       # [BN]
    nmask = (nn[None, :] < N) & elem_ok[:, None]     # [BR,BN]
    xptr = base_n[:, None] + nn[None, :] * HW
    xv = tl.load(x_ptr + xptr, mask=nmask, other=-float('inf'))
    mx = tl.max(xv, axis=1)                          # [BR]
    ev = tl.exp(xv - mx[:, None])
    denom = tl.sum(ev, axis=1)                       # [BR]
    x_elem = tl.load(x_ptr + o, mask=elem_ok, other=0.0)
    soft = tl.exp(x_elem - mx) / denom
    soft = tl.where(elem_ok, soft, 0.0)
    t = tl.load(t_ptr + o, mask=elem_ok, other=0.0)
    inter = tl.sum(soft * t, axis=0)
    card = tl.sum(soft + t, axis=0)
    dice = (2.0 * inter + smooth) / (card + smooth + eps)
    w = tl.load(w_ptr + d)
    contrib = tl.where(valid_m, w * (1.0 - dice) / Mf, 0.0)
    tl.atomic_add(out_ptr, contrib)


class MDiceLossNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-06
        self._weights = torch.tensor([0.2, 1.1, 3.0, 3.0], dtype=torch.float32)
        self._cache = {}

    def _meta(self, sizes, dev):
        key = (sizes, dev)
        c = self._cache.get(key)
        if c is not None:
            return c
        B, N, H, W = sizes
        strides = (N * H * W, H * W, W, 1)
        params = []
        Mmax = 0
        Rmax = 0
        for d in range(4):
            R = sizes[d]
            sr = strides[d]
            rem = [k for k in range(4) if k != d]
            a1, a2 = sizes[rem[1]], sizes[rem[2]]
            st0, st1, st2 = strides[rem[0]], strides[rem[1]], strides[rem[2]]
            M = sizes[rem[0]] * a1 * a2
            params.append([a1, a2, st0, st1, st2, sr, R, M])
            Mmax = max(Mmax, M)
            Rmax = max(Rmax, R)
        p_t = torch.tensor(params, device=dev, dtype=torch.int32)
        wts = self._weights.to(dev)
        BLOCK_R = triton.next_power_of_2(Rmax)
        BLOCK_N = triton.next_power_of_2(N)
        c = (p_t, wts, Mmax, BLOCK_R, BLOCK_N)
        self._cache[key] = c
        return c

    def forward(self, input: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        assert input.dim() == 4
        input = input.contiguous().float()
        target = target.contiguous().float()
        B, N, H, W = input.shape
        HW = H * W
        dev = input.device

        p_t, wts, Mmax, BLOCK_R, BLOCK_N = self._meta((B, N, H, W), dev)
        out = torch.zeros((), device=dev, dtype=torch.float32)
        _fused_kernel[(4, Mmax)](input, target, out, p_t, wts, N, HW, 1.0, self.eps,
                                 BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N, num_warps=1)
        return out
