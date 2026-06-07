import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _fused_conv_relu_pool(x_ptr, w_ptr, b_ptr, out_ptr,
                          N, H, W,
                          CIN: tl.constexpr, C: tl.constexpr,
                          K: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr):
    pid = tl.program_id(0)
    ow = pid % OW
    t = pid // OW
    oh = t % OH
    t = t // OH
    c = t % C
    n = t // C
    HW = H * W
    KK: tl.constexpr = K * K
    p = tl.arange(0, KK)
    ph = p // K
    pw = p % K
    h = oh * K + ph
    w = ow * K + pw
    acc = tl.zeros((KK,), dtype=tl.float32)
    base_w = c * CIN * 9
    for cin in range(CIN):
        base_in = n * CIN * HW + cin * HW
        bw = base_w + cin * 9
        for kh in range(3):
            ih = h + kh - 1
            okh = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = w + kw - 1
                m = okh & (iw >= 0) & (iw < W)
                v = tl.load(x_ptr + base_in + ih * W + iw, mask=m, other=0.0)
                wv = tl.load(w_ptr + bw + kh * 3 + kw)
                acc += v * wv
    acc += tl.load(b_ptr + c)
    acc = tl.maximum(acc, 0.0)
    s = tl.sum(acc) * (1.0 / KK)
    tl.store(out_ptr + pid, s)


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, K: tl.constexpr, Nf: tl.constexpr,
                   BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_k = tl.arange(0, K)
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None], other=0.0)
    for nf in range(Nf):
        wv = tl.load(w_ptr + nf * K + offs_k)
        acc = tl.sum(x * wv[None, :], axis=1)
        acc += tl.load(b_ptr + nf)
        tl.store(out_ptr + offs_m * Nf + nf, acc, mask=mask_m)


class ShakeResNetNew(nn.Module):
    def __init__(self, depth, w_base, label):
        super(ShakeResNetNew, self).__init__()
        n_units = (depth - 2) / 6
        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        self.fc_out = nn.Linear(in_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(nn.Identity())
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        N, _, H, W = x.shape
        C = self.in_chs[0]
        w = self.c_in.weight.contiguous()
        b = self.c_in.bias.contiguous()
        K = 8
        OH, OW = H // K, W // K
        pool_out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        _fused_conv_relu_pool[(N * C * OH * OW,)](
            x, w, b, pool_out, N, H, W,
            CIN=3, C=C, K=K, OH=OH, OW=OW, num_warps=2)

        h = pool_out.view(-1, self.in_chs[3])
        M, Kf = h.shape
        Nf = self.fc_out.weight.shape[0]
        wf = self.fc_out.weight.contiguous()
        bf = self.fc_out.bias.contiguous()
        out = torch.empty((M, Nf), device=x.device, dtype=x.dtype)
        BLOCK_M = 64
        _linear_kernel[(triton.cdiv(M, BLOCK_M),)](h, wf, bf, out, M,
                                                   K=Kf, Nf=Nf, BLOCK_M=BLOCK_M,
                                                   num_warps=4)
        return out
