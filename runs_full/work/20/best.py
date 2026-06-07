import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _huber_kernel(x_ptr, y_ptr, out_ptr, n_elements, inv_delta, scale, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs((x - y) * inv_delta)
    loss = tl.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    s = tl.sum(tl.where(mask, loss, 0.0), axis=0)
    tl.store(out_ptr, s * scale)


class HuberLossNew(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        x = x.contiguous()
        x_hat = x_hat.contiguous()
        n = x.numel()
        out = torch.empty((), device=x.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        scale = (self.delta * self.delta) / n
        _huber_kernel[(1,)](x, x_hat, out, n, 1.0 / self.delta, scale,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out
