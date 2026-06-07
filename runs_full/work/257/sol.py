import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _pool_kernel(x_ptr, y_ptr, HW, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * HW
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(y_ptr + row, s / HW)


@triton.jit
def _linear_kernel(in_ptr, w_ptr, b_ptr, out_ptr, OUT, K,
                   ACT: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // OUT
    o = pid % OUT
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    a = tl.load(in_ptr + n * K + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + o * K + offs, mask=mask, other=0.0)
    acc = tl.sum(a * w, axis=0) + tl.load(b_ptr + o)
    if ACT == 0:
        acc = tl.maximum(acc, 0.0)
    else:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + pid, acc)


@triton.jit
def _mul_kernel(x_ptr, scale_ptr, out_ptr, n_elements, HW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    nc = offs // HW
    s = tl.load(scale_ptr + nc, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * s, mask=mask)


class SELayerNew(nn.Module):

    def __init__(self, in_channels, reduction):
        super(SELayerNew, self).__init__()
        mid_channels = in_channels // reduction
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        n_batches, n_channels, H, W = x.size()
        x = x.contiguous()
        HW = H * W
        NC = n_batches * n_channels
        mid = self.fc1.weight.shape[0]

        pooled = torch.empty(NC, device=x.device, dtype=x.dtype)
        _pool_kernel[(NC,)](x, pooled, HW,
                            BLOCK=min(1024, triton.next_power_of_2(HW)),
                            num_warps=4)

        y1 = torch.empty(n_batches * mid, device=x.device, dtype=x.dtype)
        _linear_kernel[(n_batches * mid,)](
            pooled, self.fc1.weight, self.fc1.bias, y1, mid, n_channels,
            ACT=0, BLOCK_K=triton.next_power_of_2(n_channels), num_warps=4)

        y2 = torch.empty(NC, device=x.device, dtype=x.dtype)
        _linear_kernel[(NC,)](
            y1, self.fc2.weight, self.fc2.bias, y2, n_channels, mid,
            ACT=1, BLOCK_K=triton.next_power_of_2(mid), num_warps=4)

        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _mul_kernel[grid](x, y2, out, n_elements, HW, BLOCK=BLOCK, num_warps=4)
        return out
