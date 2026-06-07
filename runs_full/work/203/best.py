import torch
import triton
import triton.language as tl


@triton.jit
def _stable_bce_kernel(input_ptr, target_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0)
    neg_abs = -tl.abs(x)
    loss = tl.maximum(x, 0.0) - x * t + tl.log(1.0 + tl.exp(neg_abs))
    loss = tl.where(mask, loss, 0.0)
    s = tl.sum(loss, axis=0)
    tl.store(out_ptr, s / n_elements)


class StableBCELossNew(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELossNew, self).__init__()

    def forward(self, input, target):
        input = input.contiguous()
        target = target.contiguous()
        n = input.numel()
        out = torch.empty(1, device=input.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _stable_bce_kernel[(1,)](input, target, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out.reshape([])
