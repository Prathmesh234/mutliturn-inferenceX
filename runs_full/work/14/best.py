import torch
from torch import Tensor
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _patch_match(x_ptr, y_ptr, wr_ptr, wc_ptr, out_ptr,
                 C, H, W, P, threshold, inv_norm,
                 BP: tl.constexpr,
                 C_c: tl.constexpr, H_c: tl.constexpr, W_c: tl.constexpr):
    row = tl.program_id(0)   # = n * P + pi
    n = row // P
    pi = row % P
    pj = tl.arange(0, BP)
    mask = pj < P
    accx = tl.zeros((BP,), tl.float32)
    accy = tl.zeros((BP,), tl.float32)
    for c in tl.static_range(C_c):
        for r in tl.static_range(H_c):
            wr = tl.load(wr_ptr + pi * H + r)
            base = ((n * C + c) * H + r) * W
            for k in tl.static_range(W_c):
                wc = tl.load(wc_ptr + pj * W + k, mask=mask, other=0.0)
                vx = tl.load(x_ptr + base + k)
                vy = tl.load(y_ptr + base + k)
                accx += wr * wc * vx
                accy += wr * wc * vy
    bx = accx * inv_norm > threshold
    by = accy * inv_norm > threshold
    val = tl.where(bx == by, 1.0, 0.0)
    tl.store(out_ptr + row * P + pj, val, mask=mask)


class KaggleAccuracyNew(nn.Module):

    def __init__(self, threshold: 'float'=0.25, num_patches: 'int'=38, size:
        'int'=418) ->None:
        super().__init__()
        self.threshold = threshold
        self.num_patches = num_patches
        self.size = size
        self.patch_size = size // num_patches
        self.resize = nn.Upsample(size=size)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.
            patch_size)
        self._cache = {}

    def _weights(self, H, W, device):
        key = (H, W, device)
        if key in self._cache:
            return self._cache[key]
        size = self.size
        ps = self.patch_size
        P = (size - ps) // ps + 1
        ar = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, W)
        rmap = self.resize(ar)[0, 0, :, 0].round().long()
        ac = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(1, 1, H, W)
        cmap = self.resize(ac)[0, 0, 0, :].round().long()
        Wr = torch.zeros(P, H, device=device, dtype=torch.float32)
        yy = torch.arange(P * ps, device=device)
        Wr.index_put_((yy // ps, rmap[yy]), torch.ones(P * ps, device=device, dtype=torch.float32), accumulate=True)
        Wc = torch.zeros(P, W, device=device, dtype=torch.float32)
        xx = torch.arange(P * ps, device=device)
        Wc.index_put_((xx // ps, cmap[xx]), torch.ones(P * ps, device=device, dtype=torch.float32), accumulate=True)
        res = (Wr.contiguous(), Wc.contiguous(), P)
        self._cache[key] = res
        return res

    def forward(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        x = x.float().contiguous()
        y = y.float().contiguous()
        N, C, H, W = x.shape
        Wr, Wc, P = self._weights(H, W, x.device)
        ps = self.patch_size
        out = torch.empty(N * P * P, device=x.device, dtype=torch.float32)
        grid = (N * P,)
        inv_norm = 1.0 / (C * ps * ps)
        BP = triton.next_power_of_2(P)
        _patch_match[grid](x, y, Wr, Wc, out,
                           C, H, W, P, self.threshold, inv_norm, BP,
                           C_c=C, H_c=H, W_c=W, num_warps=4)
        return out.mean()
