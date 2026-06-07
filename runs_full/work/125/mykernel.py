import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(x_ptr, w_ptr, out_ptr, bias_ptr,
                 N, IC, IH, IW, OC, OH, OW, KH, KW,
                 stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                 ICg, OCg, gic, goc, K,
                 HAS_BIAS: tl.constexpr, APPLY_RELU: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    M = N * OH * OW
    m_mask = offs_m < M
    n_mask = offs_n < OCg

    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    nb = t // OH

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K
        kw = offs_k % KW
        tk = offs_k // KW
        kh = tk % KH
        ic_local = tk // KH
        ic = ic_local + gic

        ih = oh[:, None] * stride_h + kh[None, :] * dil_h - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] * dil_w - pad_w
        valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & m_mask[:, None] & k_mask[None, :]
        x_off = nb[:, None] * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        a = tl.load(x_ptr + x_off, mask=valid, other=0.0)

        oc = offs_n + goc
        w_off = oc[None, :] * K + offs_k[:, None]
        b = tl.load(w_ptr + w_off, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    if HAS_BIAS:
        oc = offs_n + goc
        bias = tl.load(bias_ptr + oc, mask=n_mask, other=0.0)
        acc += bias[None, :]
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    oc = offs_n + goc
    out_off = nb[:, None] * (OC * OH * OW) + oc[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


class BasicConvNew(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=False, bn=False, bias=True):
        super(BasicConvNew, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        bias = self.conv.bias
        N, IC, IH, IW = x.shape
        OC = self.conv.out_channels
        KH, KW = w.shape[2], w.shape[3]
        sh, sw = self.conv.stride
        ph, pw = self.conv.padding
        dh, dw = self.conv.dilation
        groups = self.conv.groups
        OH = (IH + 2 * ph - dh * (KH - 1) - 1) // sh + 1
        OW = (IW + 2 * pw - dw * (KW - 1) - 1) // sw + 1
        ICg = IC // groups
        OCg = OC // groups
        K = ICg * KH * KW
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        M = N * OH * OW
        BM, BN, BK = 16, 16, 64
        apply_relu = self.relu is not None and self.bn is None
        has_bias = bias is not None
        wflat = w.reshape(OC, K).contiguous()
        grid = (triton.cdiv(M, BM), triton.cdiv(OCg, BN))
        for g in range(groups):
            _conv_kernel[grid](
                x, wflat, out, bias if has_bias else x,
                N, IC, IH, IW, OC, OH, OW, KH, KW,
                sh, sw, ph, pw, dh, dw,
                ICg, OCg, g * ICg, g * OCg, K,
                HAS_BIAS=has_bias, APPLY_RELU=apply_relu,
                BM=BM, BN=BN, BK=BK, num_warps=1, num_stages=1)
        if self.bn is not None:
            out = self.bn(out)
            if self.relu is not None:
                out = self.relu(out)
        return out
