import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, ginv_ptr, beta_ptr, out_ptr,
                  N, ID, IH, IW, OD, OH, OW, SPATIAL, eps,
                  IC: tl.constexpr, OC: tl.constexpr,
                  KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                  STRIDE: tl.constexpr, PAD: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    oc = pid % OC
    n = pid // OC
    offs = tl.arange(0, BLOCK)
    mask = offs < SPATIAL
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    od = t // OH
    acc = tl.zeros((BLOCK,), tl.float32)
    for ic in range(IC):
        for kd in range(KD):
            id_ = od * STRIDE + kd - PAD
            for kh in range(KH):
                ih = oh * STRIDE + kh - PAD
                for kw in range(KW):
                    iw = ow * STRIDE + kw - PAD
                    valid = (id_ >= 0) & (id_ < ID) & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                    x_idx = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                    xv = tl.load(x_ptr + x_idx, mask=mask & valid, other=0.0)
                    w_idx = (((oc * IC + ic) * KD + kd) * KH + kh) * KW + kw
                    wv = tl.load(w_ptr + w_idx)
                    acc += xv * wv
    acc += tl.load(b_ptr + oc)
    cnt = SPATIAL.to(tl.float32)
    mean = tl.sum(acc, axis=0) / cnt
    xc = tl.where(mask, acc - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / cnt
    inv = 1.0 / tl.sqrt(var + eps)
    g = tl.load(ginv_ptr + oc)
    bt = tl.load(beta_ptr + oc)
    y = xc * inv * g + bt
    y = tl.maximum(y, 0.0)
    base = pid * SPATIAL
    tl.store(out_ptr + base + offs, y, mask=mask)


class DownsampleNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        squeeze = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze = True
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        w = self.conv3d.weight.contiguous()
        b = self.conv3d.bias.contiguous()
        OC = w.shape[0]
        KD, KH, KW = 3, 3, 3
        STRIDE, PAD = 2, 1
        OD = (ID + 2 * PAD - KD) // STRIDE + 1
        OH = (IH + 2 * PAD - KH) // STRIDE + 1
        OW = (IW + 2 * PAD - KW) // STRIDE + 1
        SPATIAL = OD * OH * OW
        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(SPATIAL)
        _fused_kernel[(N * OC,)](x, w, b, self.bn1.weight, self.bn1.bias, out,
                                 N, ID, IH, IW, OD, OH, OW, SPATIAL, self.bn1.eps,
                                 IC, OC, KD, KH, KW, STRIDE, PAD, BLOCK=BLOCK, num_warps=1)
        if squeeze:
            out = out.squeeze(0)
        return out
