import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sce_kernel(x_ptr, t_ptr, out_ptr, R, C, inner, scale,
                BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < R
    o = rows // inner
    i = rows % inner
    base = o * (C * inner) + i
    carange = tl.arange(0, BLOCK_C)
    c_mask = carange < C
    mask = row_mask[:, None] & c_mask[None, :]
    ptr = base[:, None] + carange[None, :] * inner

    x = tl.load(x_ptr + ptr, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    x_centered = x - m[:, None]
    e = tl.exp(tl.where(mask, x_centered, -float('inf')))
    s = tl.sum(e, axis=1)
    logsm = x_centered - tl.log(s)[:, None]

    t = tl.load(t_ptr + ptr, mask=mask, other=0.0)
    row_loss = tl.sum(tl.where(mask, -t * logsm, 0.0), axis=1)
    total = tl.sum(tl.where(row_mask, row_loss, 0.0))
    tl.store(out_ptr, total * scale)


class SmoothCrossEntropyLossNew(nn.Module):

    def __init__(self, label_smoothing=0.0, size_average=True):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=input.size(-1)).float()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            target = target * (1.0 - s_by_c) + s_by_c

        input = input.contiguous()
        target = target.contiguous().to(input.dtype)

        C = input.size(1)
        outer = input.size(0)
        inner = input.numel() // (outer * C)
        R = outer * inner

        out = torch.empty((), dtype=torch.float32, device=input.device)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_R = triton.next_power_of_2(R)
        scale = 1.0 / R if self.size_average else 1.0
        _sce_kernel[(1,)](input, target, out, R, C, inner, scale,
                          BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=2)
        return out.to(input.dtype)
