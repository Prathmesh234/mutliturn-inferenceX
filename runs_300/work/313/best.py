import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _catconv_kernel(x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
                    P, OC,
                    IC1: tl.constexpr, IC2: tl.constexpr,
                    BLOCK_OC: tl.constexpr, BLOCK_P: tl.constexpr):
    n = tl.program_id(0)
    IC = IC1 + IC2

    offs_oc = tl.arange(0, BLOCK_OC)
    offs_p = tl.arange(0, BLOCK_P)
    moc = offs_oc < OC
    mp = offs_p < P

    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    x2_base = n * IC2 * P
    for ic in tl.static_range(IC2):
        w = tl.load(w_ptr + offs_oc * IC + ic, mask=moc, other=0.0)
        xv = tl.load(x2_ptr + x2_base + ic * P + offs_p, mask=mp, other=0.0)
        acc += w[:, None] * xv[None, :]

    x1_base = n * IC1 * P
    for ic in tl.static_range(IC1):
        w = tl.load(w_ptr + offs_oc * IC + IC2 + ic, mask=moc, other=0.0)
        xv = tl.load(x1_ptr + x1_base + ic * P + offs_p, mask=mp, other=0.0)
        acc += w[:, None] * xv[None, :]

    b = tl.load(b_ptr + offs_oc, mask=moc, other=0.0)
    acc += b[:, None]

    out_base = n * OC * P
    out_ptr_2d = out_ptr + out_base + offs_oc[:, None] * P + offs_p[None, :]
    tl.store(out_ptr_2d, acc, mask=moc[:, None] & mp[None, :])


class CatConvNew(nn.Module):
    def __init__(self, in_kernels_1, in_kernels_2, kernels):
        super(CatConvNew, self).__init__()
        self.conv = nn.Conv2d(in_kernels_1 + in_kernels_2, kernels,
            kernel_size=1, bias=True)
        self.in_kernels_1 = in_kernels_1
        self.in_kernels_2 = in_kernels_2

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        N, _, H, W = x1.shape
        P = H * W
        OC = self.conv.out_channels
        IC1 = self.in_kernels_1
        IC2 = self.in_kernels_2
        out = torch.empty((N, OC, H, W), device=x1.device, dtype=x1.dtype)
        weight = self.conv.weight.reshape(OC, IC1 + IC2)
        bias = self.conv.bias
        BLOCK_OC = triton.next_power_of_2(OC)
        BLOCK_P = triton.next_power_of_2(P)
        grid = (N,)
        _catconv_kernel[grid](x1, x2, weight, bias, out,
                              P, OC, IC1, IC2, BLOCK_OC, BLOCK_P,
                              num_warps=4)
        return out
