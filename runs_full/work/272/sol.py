import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ghmr_stats_kernel(pred_ptr, target_ptr, lw_ptr, counts_ptr, sumloss_ptr,
                       n_elements, mu, bins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    diff = p - t
    d2 = diff * diff
    root = tl.sqrt(d2 + mu * mu)
    loss_elem = root - mu
    g = tl.abs(diff / root)

    valid = mask & (lw > 0.0)
    bin_idx = (g * bins).to(tl.int32)
    bin_idx = tl.minimum(bin_idx, bins - 1)

    tl.atomic_add(counts_ptr + bin_idx, 1.0, mask=valid)
    tl.atomic_add(sumloss_ptr + bin_idx, loss_elem, mask=valid)


@triton.jit
def _ghmr_reduce_kernel(counts_ptr, sumloss_ptr, out_ptr, bins, loss_weight,
                        BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < bins
    counts = tl.load(counts_ptr + offs, mask=mask, other=0.0)
    sumloss = tl.load(sumloss_ptr + offs, mask=mask, other=0.0)
    nonempty = counts > 0.0
    n = tl.sum(tl.where(nonempty, 1.0, 0.0))
    per = tl.where(nonempty, sumloss / counts, 0.0)
    s = tl.sum(per)
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

        counts = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        sumloss = torch.zeros(bins, device=pred.device, dtype=torch.float32)

        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _ghmr_stats_kernel[grid](pred, target, label_weight, counts, sumloss,
                                 n_elements, mu, bins, BLOCK_SIZE=BLOCK,
                                 num_warps=4)

        if mmt > 0:
            # sequential per-bin moving-average update on tiny (bins,) state
            n = 0
            total = torch.zeros((), device=pred.device, dtype=torch.float32)
            cnt_cpu = counts.tolist()
            for i in range(bins):
                num_in_bin = cnt_cpu[i]
                if num_in_bin > 0:
                    n += 1
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    total = total + sumloss[i] / self.acc_sum[i]
            if n > 0:
                result = total / n
            else:
                result = total
            return result * self.loss_weight

        out = torch.empty((), device=pred.device, dtype=torch.float32)
        bpow = 1
        while bpow < bins:
            bpow *= 2
        _ghmr_reduce_kernel[(1,)](counts, sumloss, out, bins, self.loss_weight,
                                  BLOCK_SIZE=bpow, num_warps=1)
        return out
