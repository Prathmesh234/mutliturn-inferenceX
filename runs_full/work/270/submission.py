import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _psnr_kernel(x_ptr, y_ptr, out_ptr, n_elements, inv_n, max_val_sq, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, n_elements, BLOCK_SIZE):
        o = start + offs
        mask = o < n_elements
        x = tl.load(x_ptr + o, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + o, mask=mask, other=0.0).to(tl.float32)
        d = x - y
        acc += d * d
    sse = tl.sum(acc)
    mse = sse * inv_n
    loss = -10.0 * (tl.log(max_val_sq / mse) * 0.43429448190325176)
    tl.store(out_ptr, loss)


class PSNRLossNew(nn.Module):
    def __init__(self, max_val: float) -> None:
        super(PSNRLossNew, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input.contiguous()
        y = target.contiguous()
        n = x.numel()
        out = torch.empty((), device=x.device, dtype=torch.float32)
        BLOCK_SIZE = 256
        _psnr_kernel[(1,)](x, y, out, n, 1.0 / n, float(self.max_val) ** 2,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.to(input.dtype)
