import torch
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _time_encode_kernel(t_ptr, w_ptr, b_ptr, out_ptr, n_rows, dim, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < dim
    tval = tl.load(t_ptr + row)
    w = tl.load(w_ptr + offs_d, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_d, mask=mask, other=0.0)
    out = tl.cos(tval * w + b)
    tl.store(out_ptr + row * dim + offs_d, out, mask=mask)


class TimeEncodeNew(torch.nn.Module):

    def __init__(self, dimension):
        super(TimeEncodeNew, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.
            linspace(0, 9, dimension)).float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        orig_shape = t.shape
        t = t.contiguous()
        n_rows = t.numel()
        dim = self.dimension
        out = torch.empty((n_rows, dim), device=t.device, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(dim)
        grid = (n_rows,)
        _time_encode_kernel[grid](
            t.view(-1), self.w.weight.view(-1), self.w.bias, out,
            n_rows, dim, BLOCK_D=BLOCK_D, num_warps=1)
        return out.view(*orig_shape, dim)
