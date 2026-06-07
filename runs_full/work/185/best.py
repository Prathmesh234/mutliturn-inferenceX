import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _focal_kernel(logits_ptr, targets_ptr, out_ptr, acc_ptr, n,
                  gamma, alpha, threshold, scale,
                  REDUCED: tl.constexpr, USE_ALPHA: tl.constexpr,
                  STORE: tl.constexpr, ACC: tl.constexpr, SINGLE: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(logits_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(targets_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    bce = tl.maximum(x, 0.0) - x * t + tl.log(1.0 + tl.exp(-ax))
    logpt = -bce
    pt = tl.exp(logpt)

    if REDUCED:
        fr = tl.exp(gamma * tl.log((1.0 - pt) / threshold))
        fr = tl.where(pt < threshold, 1.0, fr)
        loss = -fr * logpt
    else:
        loss = -tl.exp(gamma * tl.log(1.0 - pt)) * logpt
        if USE_ALPHA:
            loss = loss * (alpha * t + (1.0 - alpha) * (1.0 - t))

    if STORE:
        tl.store(out_ptr + offs, loss, mask=mask)
    if ACC:
        block_sum = tl.sum(tl.where(mask, loss * scale, 0.0))
        if SINGLE:
            tl.store(acc_ptr, block_sum)
        else:
            tl.atomic_add(acc_ptr, block_sum)


class FocalLossBinaryNew(nn.Module):

    def __init__(self, ignore: int = None, reduced: bool = False, gamma: float = 2.0,
                 alpha: float = 0.25, threshold: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.ignore = ignore
        self.reduced = reduced
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.view(-1)
        logits = logits.view(-1)
        if self.ignore is not None:
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        logits = logits.contiguous()
        targets = targets.contiguous().to(logits.dtype)
        n = logits.numel()
        reduction = self.reduction
        use_alpha = (not self.reduced) and (self.alpha is not None)
        alpha = self.alpha if self.alpha is not None else 0.0

        if reduction == 'none':
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            out = torch.empty_like(logits)
            _focal_kernel[grid](logits, targets, out, out, n,
                                self.gamma, alpha, self.threshold, 1.0,
                                self.reduced, use_alpha, True, False, False,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
            return out

        if reduction in ('mean', 'sum'):
            scale = (1.0 / n) if reduction == 'mean' else 1.0
            # single-block fast path avoids zeroing + atomics
            BLOCK_SIZE = 1 << max(0, (n - 1).bit_length()) if n > 0 else 1
            if BLOCK_SIZE <= 2048:
                acc = torch.empty((), device=logits.device, dtype=torch.float32)
                _focal_kernel[(1,)](logits, targets, logits, acc, n,
                                    self.gamma, alpha, self.threshold, scale,
                                    self.reduced, use_alpha, False, True, True,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
                return acc.to(logits.dtype)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            acc = torch.zeros((), device=logits.device, dtype=torch.float32)
            _focal_kernel[grid](logits, targets, logits, acc, n,
                                self.gamma, alpha, self.threshold, scale,
                                self.reduced, use_alpha, False, True, False,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
            return acc.to(logits.dtype)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        out = torch.empty_like(logits)
        _focal_kernel[grid](logits, targets, out, out, n,
                            self.gamma, alpha, self.threshold, 1.0,
                            self.reduced, use_alpha, True, False, False,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.sum(0)
