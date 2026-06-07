import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reorg_kernel(in_ptr, out_ptr, n_elements,
                  C, H, W, OC, Ho, Wo, s,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    w2 = idx % Wo
    t = idx // Wo
    h2 = t % Ho
    t = t // Ho
    oc = t % OC
    b = t // OC

    c = oc % C
    j = oc // C
    iw = j % s
    ih = j // s

    h = h2 * s + ih
    w = w2 * s + iw
    in_off = ((b * C + c) * H + h) * W + w

    val = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + idx, val, mask=mask)


class ReorgNew(nn.Module):
    def __init__(self, stride=2):
        super(ReorgNew, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.dim() == 4
        B, C, H, W = x.shape
        assert H % stride == 0
        assert W % stride == 0
        x = x.contiguous()
        Ho = H // stride
        Wo = W // stride
        OC = C * stride * stride
        out = torch.empty((B, OC, Ho, Wo), device=x.device, dtype=x.dtype)
        n = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _reorg_kernel[grid](x, out, n, C, H, W, OC, Ho, Wo, stride,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
