import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _zeropad1d_kernel(x_ptr, o_ptr, n_out, W_in, W_out, pad_left,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    row = offs // W_out
    col = offs % W_out
    col_in = col - pad_left
    in_range = (col_in >= 0) & (col_in < W_in)
    in_off = row * W_in + col_in
    val = tl.load(x_ptr + in_off, mask=mask & in_range, other=0.0)
    tl.store(o_ptr + offs, val, mask=mask)


class ZeroPad1dNew(nn.Module):

    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        W_in = x.shape[-1]
        W_out = W_in + self.pad_left + self.pad_right
        out_shape = list(x.shape[:-1]) + [W_out]
        x = x.contiguous()
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        n_out = out.numel()
        BLOCK_SIZE = triton.next_power_of_2(n_out)
        grid = (1,)
        _zeropad1d_kernel[grid](x, out, n_out, W_in, W_out, self.pad_left,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
