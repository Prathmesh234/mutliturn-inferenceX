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
def _hist_kernel(pred_ptr, target_ptr, lw_ptr, hist_ptr, n_elements, bins,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(pred)
    g = tl.abs(s - target)
    b = (g * bins).to(tl.int32)
    b = tl.minimum(b, bins - 1)
    b = tl.maximum(b, 0)
    valid = (lw > 0.0) & mask
    tl.atomic_add(hist_ptr + b, 1.0, mask=valid)


@triton.jit
def _loss_kernel(pred_ptr, target_ptr, lw_ptr, wbin_ptr, loss_ptr, n_elements,
                 bins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    lw = tl.load(lw_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(pred)
    g = tl.abs(s - target)
    b = (g * bins).to(tl.int32)
    b = tl.minimum(b, bins - 1)
    b = tl.maximum(b, 0)
    valid = (lw > 0.0) & mask
    w = tl.load(wbin_ptr + b, mask=valid, other=0.0)
    # BCE with logits: max(x,0) - x*z + log(1+exp(-|x|))
    bce = tl.maximum(pred, 0.0) - pred * target + tl.log(1.0 + tl.exp(-tl.abs(pred)))
    contrib = tl.where(valid, w * bce, 0.0)
    partial = tl.sum(contrib)
    tl.atomic_add(loss_ptr, partial)


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
        hist = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _hist_kernel[grid](pred, target, label_weight, hist, n, bins,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        tot = max(hist.sum().item(), 1.0)
        nz = hist > 0
        n_nonempty = int(nz.sum().item())

        if mmt > 0:
            self.acc_sum = torch.where(nz, mmt * self.acc_sum +
                                       (1 - mmt) * hist, self.acc_sum)
            denom = self.acc_sum
        else:
            denom = hist

        wbin = torch.zeros(bins, device=pred.device, dtype=torch.float32)
        if n_nonempty > 0:
            wbin = torch.where(nz, tot / denom / n_nonempty, wbin)

        loss = torch.zeros(1, device=pred.device, dtype=torch.float32)
        _loss_kernel[grid](pred, target, label_weight, wbin, loss, n, bins,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        loss = loss / tot
        return loss[0] * self.loss_weight
