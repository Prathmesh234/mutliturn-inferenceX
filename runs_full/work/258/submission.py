import torch
import torch.nn as nn
import triton
import triton.language as tl


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels),
        as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0),
        label_channels)
    return bin_labels, bin_label_weights


@triton.jit
def _fused_kernel(pred_ptr, target_ptr, lw_ptr, out_ptr, n_elements,
                  loss_weight, BINS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(pred)
    g = tl.abs(s - target)
    b = (g * BINS).to(tl.int32)
    b = tl.minimum(b, BINS - 1)
    b = tl.maximum(b, 0)
    valid = (lw > 0.0) & mask
    valid_f = valid.to(tl.float32)
    tot = tl.sum(valid_f)
    bce = tl.maximum(pred, 0.0) - pred * target + tl.log(1.0 + tl.exp(-tl.abs(pred)))

    w_elem = tl.zeros((BLOCK,), tl.float32)
    n_nonempty = 0.0
    for i in range(BINS):
        in_bin = valid & (b == i)
        cnt = tl.sum(in_bin.to(tl.float32))
        has = cnt > 0.0
        n_nonempty += has.to(tl.float32)
        wi = tl.where(has, tot / cnt, 0.0)
        w_elem = tl.where(in_bin, wi, w_elem)

    nsafe = tl.maximum(n_nonempty, 1.0)
    totsafe = tl.maximum(tot, 1.0)
    w_elem = w_elem / nsafe
    contrib = tl.where(valid, w_elem * bce, 0.0)
    loss = tl.sum(contrib) / totsafe * loss_weight
    tl.store(out_ptr + offs, loss, mask=offs == 0)


@triton.jit
def _hist_kernel(pred_ptr, target_ptr, lw_ptr, hist_ptr, n_elements, bins,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0)
    g = tl.abs(tl.sigmoid(pred) - target)
    b = (g * bins).to(tl.int32)
    b = tl.minimum(tl.maximum(b, 0), bins - 1)
    valid = (lw > 0.0) & mask
    tl.atomic_add(hist_ptr + b, 1.0, mask=valid)


@triton.jit
def _reduce_kernel(hist_ptr, wbin_ptr, tot_ptr, bins, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < bins
    h = tl.load(hist_ptr + offs, mask=mask, other=0.0)
    tot = tl.maximum(tl.sum(h), 1.0)
    nz = h > 0.0
    nsafe = tl.maximum(tl.sum(nz.to(tl.float32)), 1.0)
    wbin = tl.where(nz, tot / h / nsafe, 0.0)
    tl.store(wbin_ptr + offs, wbin, mask=mask)
    tl.store(tot_ptr + offs, tot, mask=offs == 0)


@triton.jit
def _loss_kernel(pred_ptr, target_ptr, lw_ptr, wbin_ptr, loss_ptr, n_elements,
                 bins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0)
    g = tl.abs(tl.sigmoid(pred) - target)
    b = (g * bins).to(tl.int32)
    b = tl.minimum(tl.maximum(b, 0), bins - 1)
    valid = (lw > 0.0) & mask
    w = tl.load(wbin_ptr + b, mask=valid, other=0.0)
    bce = tl.maximum(pred, 0.0) - pred * target + tl.log(1.0 + tl.exp(-tl.abs(pred)))
    contrib = tl.where(valid, w * bce, 0.0)
    tl.atomic_add(loss_ptr, tl.sum(contrib))


class GHMCNew(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMCNew, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-06
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target,
                label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        pred = pred.float().contiguous()
        target = target.contiguous()
        label_weight = label_weight.contiguous()
        mmt = self.momentum
        bins = self.bins
        n = pred.numel()

        if mmt == 0 and n <= 16384:
            BLOCK = triton.next_power_of_2(n)
            nw = 4 if BLOCK >= 1024 else 2
            out = torch.empty(1, device=pred.device, dtype=torch.float32)
            _fused_kernel[(1,)](pred, target, label_weight, out, n,
                                float(self.loss_weight), BINS=bins, BLOCK=BLOCK,
                                num_warps=nw)
            return out[0]

        # general fallback
        hist = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _hist_kernel[grid](pred, target, label_weight, hist, n, bins,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        wbin = torch.empty(bins, device=pred.device, dtype=torch.float32)
        tot_t = torch.empty(1, device=pred.device, dtype=torch.float32)
        if mmt > 0:
            tot = torch.clamp(hist.sum(), min=1.0)
            nz = hist > 0
            n_safe = torch.clamp(nz.sum().to(torch.float32), min=1.0)
            self.acc_sum = torch.where(nz, mmt * self.acc_sum +
                                       (1 - mmt) * hist, self.acc_sum)
            wbin = torch.where(nz, tot / self.acc_sum / n_safe,
                               torch.zeros_like(hist))
            tot_t = tot.reshape(1)
        else:
            BR = triton.next_power_of_2(bins)
            _reduce_kernel[(1,)](hist, wbin, tot_t, bins, BLOCK=BR, num_warps=1)
        loss = torch.zeros(1, device=pred.device, dtype=torch.float32)
        _loss_kernel[grid](pred, target, label_weight, wbin, loss, n, bins,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return (loss[0] / tot_t[0]) * self.loss_weight
