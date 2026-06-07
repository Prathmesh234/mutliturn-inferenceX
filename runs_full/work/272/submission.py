import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ghmr_fused_kernel(pred_ptr, target_ptr, lw_ptr, out_ptr,
                       n_elements, mu, loss_weight,
                       BINS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    diff = p - t
    root = tl.sqrt(diff * diff + mu * mu)
    loss_elem = root - mu
    g = tl.abs(diff / root)

    valid = mask & (lw > 0.0)
    bin_idx = tl.minimum((g * BINS).to(tl.int32), BINS - 1)

    n = 0.0
    s = 0.0
    for i in tl.static_range(BINS):
        in_bin = valid & (bin_idx == i)
        cnt = tl.sum(tl.where(in_bin, 1.0, 0.0))
        sl = tl.sum(tl.where(in_bin, loss_elem, 0.0))
        has = cnt > 0.0
        n += tl.where(has, 1.0, 0.0)
        s += tl.where(has, sl / tl.where(has, cnt, 1.0), 0.0)

    result = tl.where(n > 0.0, s / n, 0.0) * loss_weight
    tl.store(out_ptr, result)


class GHMRNew(nn.Module):
    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        super(GHMRNew, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] = 1000.0
        self.momentum = momentum
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        mu = self.mu
        bins = self.bins
        mmt = self.momentum

        pred = pred.contiguous()
        target = target.contiguous()
        label_weight = label_weight.contiguous()
        n_elements = pred.numel()

        bpow = 1
        while bpow < n_elements:
            bpow *= 2

        if mmt == 0 and bpow <= 4096:
            out = torch.empty((), device=pred.device, dtype=torch.float32)
            _ghmr_fused_kernel[(1,)](pred, target, label_weight, out,
                                     n_elements, mu, self.loss_weight,
                                     BINS=bins, BLOCK_SIZE=bpow,
                                     num_warps=8)
            return out

        # fallback (large n or momentum) -- compute stats then reduce in torch
        counts = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        sumloss = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        diff = pred - target
        root = torch.sqrt(diff * diff + mu * mu)
        loss_elem = (root - mu).flatten()
        g = torch.abs(diff / root).flatten()
        valid = (label_weight.flatten() > 0)
        bin_idx = torch.clamp((g * bins).to(torch.int32), max=bins - 1)
        for i in range(bins):
            m = valid & (bin_idx == i)
            counts[i] = m.sum()
            sumloss[i] = loss_elem[m].sum()
        n = 0
        total = torch.zeros((), device=pred.device, dtype=torch.float32)
        for i in range(bins):
            c = counts[i].item()
            if c > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * c
                    total = total + sumloss[i] / self.acc_sum[i]
                else:
                    total = total + sumloss[i] / counts[i]
        result = total / n if n > 0 else total
        return result * self.loss_weight
