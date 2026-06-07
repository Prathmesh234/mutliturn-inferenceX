import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gap_clip_kernel(x_ptr, out_ptr, spatial, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < spatial
    x = tl.load(x_ptr + pid * spatial + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    m = s / spatial
    m = tl.minimum(tl.maximum(m, 0.0), 1.0)
    tl.store(out_ptr + pid, m)


class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class ClipGlobalAvgPoolNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avgpool = FastGlobalAvgPool()

    def forward(self, x):
        N, C = x.size(0), x.size(1)
        nrows = N * C
        spatial = x.numel() // nrows
        out = torch.empty((nrows,), device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(spatial)
        _gap_clip_kernel[(nrows,)](x, out, spatial, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out.view(N, C, 1, 1)
