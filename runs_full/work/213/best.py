import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                    HW,
                    IC: tl.constexpr, OC: tl.constexpr,
                    IC_P: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0)
    pid = tl.program_id(1)
    hw = pid * BLOCK + tl.arange(0, BLOCK)
    mask = hw < HW
    icr = tl.arange(0, IC_P)
    in_base = n * (IC * HW) + hw
    x_off = in_base[:, None] + icr[None, :] * HW
    x_mask = mask[:, None] & (icr[None, :] < IC)
    x2d = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
    out_base = n * (OC * HW) + hw
    for oc in tl.static_range(OC):
        wrow = tl.load(w_ptr + oc * IC + icr, mask=icr < IC, other=0.0)
        acc = tl.sum(x2d * wrow[None, :], axis=1) + tl.load(b_ptr + oc)
        tl.store(out_ptr + out_base + oc * HW, acc, mask=mask)


class conv2New(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, is_deconv=False,
                 is_batchnorm=False, *args, **kwargs):
        super(conv2New, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.final = nn.Conv2d(self.in_channels, num_classes, 1)

    def forward(self, inputs):
        inputs = inputs.contiguous()
        N, IC, H, W = inputs.shape
        OC = self.final.weight.shape[0]
        HW = H * W
        out = torch.empty((N, OC, H, W), device=inputs.device, dtype=inputs.dtype)
        w = self.final.weight.reshape(OC, IC).contiguous()
        b = self.final.bias.contiguous()
        BLOCK = 256
        IC_P = triton.next_power_of_2(IC)
        grid = (N, triton.cdiv(HW, BLOCK))
        _conv1x1_kernel[grid](inputs, w, b, out, HW,
                              IC=IC, OC=OC, IC_P=IC_P,
                              BLOCK=BLOCK, num_warps=4)
        return out
