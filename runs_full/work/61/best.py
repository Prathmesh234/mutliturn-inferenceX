import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elem
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _maxpool2d_kernel(x_ptr, out_ptr, n_out, C, H, W, OH, OW, K,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    c = t % C
    b = t // C
    ih0 = oh * K
    iw0 = ow * K
    base = ((b * C + c) * H) * W
    acc = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    for i in range(K):
        for j in range(K):
            ih = ih0 + i
            iw = iw0 + j
            valid = mask & (ih < H) & (iw < W)
            ptr = x_ptr + base + ih * W + iw
            v = tl.load(ptr, mask=valid, other=-float('inf'))
            acc = tl.maximum(acc, v)
    tl.store(out_ptr + offs, acc, mask=mask)


class MultiLevelPoolingNew(nn.Module):

    def __init__(self, levels=[1, 2, 4]):
        super(MultiLevelPoolingNew, self).__init__()
        self.Pools = nn.ModuleList([nn.MaxPool2d(i) for i in levels])
        self.levels = list(levels)

    def forward(self, x):
        assert len(x.size()) == 4, '输入形状不满足(n,c,w,w)'
        n, c, H, W = x.shape
        K = self.levels[0]
        x = x.contiguous()
        if K == 1:
            out = torch.empty_like(x)
            n_elem = x.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
            _copy_kernel[grid](x, out, n_elem, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
            return out.view(n, c, -1)
        OH = H // K
        OW = W // K
        out = torch.empty((n, c, OH, OW), device=x.device, dtype=x.dtype)
        n_out = n * c * OH * OW
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_out, BLOCK_SIZE),)
        _maxpool2d_kernel[grid](x, out, n_out, c, H, W, OH, OW, K,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.view(n, c, -1)
