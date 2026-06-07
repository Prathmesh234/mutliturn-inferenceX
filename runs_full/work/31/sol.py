import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _deconv_kernel(
    x_ptr, w_ptr, o_ptr,
    N, CI, IH, IW, CO, OH, OW, KH, KW,
    stride, pad, slope, FUSE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    M = N * OH * OW
    m_mask = offs_m < M
    n_mask = offs_n < CO

    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    nb = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ihn = oh + pad - kh
            iwn = ow + pad - kw
            ih = ihn // stride
            iw = iwn // stride
            valid = m_mask & (ihn % stride == 0) & (iwn % stride == 0)
            valid = valid & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            row_base = nb * (CI * IH * IW) + ih * IW + iw
            wbase = kh * KW + kw
            for k0 in range(0, CI, BLOCK_K):
                ci = k0 + tl.arange(0, BLOCK_K)
                ci_mask = ci < CI
                a_off = row_base[:, None] + ci[None, :] * (IH * IW)
                a_mask = valid[:, None] & ci_mask[None, :]
                a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0)
                w_off = ci[:, None] * (CO * KH * KW) + offs_n[None, :] * (KH * KW) + wbase
                w_mask = ci_mask[:, None] & n_mask[None, :]
                w = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
                acc += tl.dot(a, w)

    if FUSE:
        acc = tl.where(acc > 0, acc, acc * slope)

    out_row = nb * (CO * OH * OW) + oh * OW + ow
    o_off = out_row[:, None] + offs_n[None, :] * (OH * OW)
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(o_ptr + o_off, acc, mask=o_mask)


def deconv2d(x, weight, stride, pad, fuse_lrelu, slope=0.01,
             BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, num_warps=4):
    N, CI, IH, IW = x.shape
    CI2, CO, KH, KW = weight.shape
    OH = (IH - 1) * stride - 2 * pad + KH
    OW = (IW - 1) * stride - 2 * pad + KW
    out = torch.empty((N, CO, OH, OW), device=x.device, dtype=x.dtype)
    M = N * OH * OW
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(CO, BLOCK_N))
    _deconv_kernel[grid](
        x, weight, out,
        N, CI, IH, IW, CO, OH, OW, KH, KW,
        stride, pad, slope, fuse_lrelu,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=1,
    )
    return out


class DecoderNew(nn.Module):
    def __init__(self, z_dim):
        super(DecoderNew, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, 128, kernel_size=4, stride=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = deconv2d(x, self.deconv1.weight, 1, 0, True, BLOCK_M=16, BLOCK_N=32, BLOCK_K=16, num_warps=2)
        x = deconv2d(x, self.deconv2.weight, 2, 1, True, BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, num_warps=2)
        x = deconv2d(x, self.deconv3.weight, 2, 1, False, BLOCK_M=16, BLOCK_N=16, BLOCK_K=64, num_warps=2)
        return x
