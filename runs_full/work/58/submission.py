import torch
import triton
import triton.language as tl


@triton.jit
def _to_half_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x.to(tl.float16), mask=mask)


class ToHalfNew(torch.nn.Module):
    def forward(self, tensor):
        x = tensor.contiguous()
        out = torch.empty_like(x, dtype=torch.float16)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _to_half_kernel[(1,)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
