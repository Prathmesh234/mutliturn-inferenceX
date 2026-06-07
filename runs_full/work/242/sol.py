import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _contract_kernel(x_ptr, out_ptr, total,
                     N, C, Hs, Ws, s,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    OC = C * s * s
    H = Hs * s
    W = Ws * s

    wp = offs % Ws
    t1 = offs // Ws
    hp = t1 % Hs
    t2 = t1 // Hs
    oc = t2 % OC
    n = t2 // OC

    c = oc % C
    t3 = oc // C
    sw = t3 % s
    sh = t3 // s

    h = hp * s + sh
    w = wp * s + sw
    in_idx = ((n * C + c) * H + h) * W + w

    val = tl.load(x_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


class ContractNew(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.contiguous()
        Hs = H // s
        Ws = W // s
        out = torch.empty((N, C * s * s, Hs, Ws), device=x.device, dtype=x.dtype)
        total = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        _contract_kernel[grid](x, out, total, N, C, Hs, Ws, s,
                               BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
