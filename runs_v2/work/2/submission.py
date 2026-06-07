import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _customize_kernel(x_ptr, scale_ptr, bias_ptr, out_ptr,
                      N, C, INNER, W,
                      BLOCK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    offs_p = pid_p * BLOCK + tl.arange(0, BLOCK)
    mask = offs_p < INNER
    base = pid_n * C * INNER + offs_p

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(C):
        x = tl.load(x_ptr + base + c * INNER, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    inv = 1.0 / tl.sqrt(acc)

    w = offs_p % W
    s = tl.load(scale_ptr + w, mask=mask, other=0.0).to(tl.float32) * inv
    b = tl.load(bias_ptr + w, mask=mask, other=0.0).to(tl.float32)
    for c in range(C):
        ptr = base + c * INNER
        x = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + ptr, x * s + b, mask=mask)


class CustomizeLayerNew(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))

    def forward(self, x):
        shape = x.shape
        N = shape[0]
        C = shape[1]
        W = shape[-1]
        INNER = 1
        for d in shape[2:]:
            INNER *= d
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK = 256
        grid = (N, triton.cdiv(INNER, BLOCK))
        _customize_kernel[grid](xc, self.scale, self.bias, out,
                                N, C, INNER, W, BLOCK=BLOCK, num_warps=4)
        return out.view(shape)

    def __repr__(self):
        return 'CustomizedLayer(in_dim=%d)' % self.in_dim
