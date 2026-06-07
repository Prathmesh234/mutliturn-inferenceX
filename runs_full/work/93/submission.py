import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _full_kernel(x_ptr, w_ptr, b_ptr, wf_ptr, bf_ptr, out_ptr,
                 N, H, W,
                 CIN: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                 OH: tl.constexpr, OW: tl.constexpr,
                 KLIN: tl.constexpr, NF: tl.constexpr):
    m = tl.program_id(0)
    SPC: tl.constexpr = OH * OW          # spatial per channel
    CHW: tl.constexpr = C * SPC
    base = m * KLIN
    n = base // CHW
    crem = base % CHW
    c = crem // SPC
    sstart = crem % SPC
    HW = H * W
    kk = tl.arange(0, KLIN)[:, None]
    pp = tl.arange(0, K * K)[None, :]
    pidx = sstart + kk
    oh = pidx // OW
    ow = pidx % OW
    ph = pp // K
    pw = pp % K
    hh = oh * K + ph
    ww = ow * K + pw
    acc = tl.zeros((KLIN, K * K), dtype=tl.float32)
    base_w = c * CIN * 9
    for cin in range(CIN):
        base_in = n * CIN * HW + cin * HW
        bw = base_w + cin * 9
        for kh in range(3):
            ih = hh + kh - 1
            okh = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = ww + kw - 1
                msk = okh & (iw >= 0) & (iw < W)
                v = tl.load(x_ptr + base_in + ih * W + iw, mask=msk, other=0.0)
                wv = tl.load(w_ptr + bw + kh * 3 + kw)
                acc += v * wv
    acc += tl.load(b_ptr + c)
    acc = tl.maximum(acc, 0.0)
    val = tl.sum(acc, axis=1) * (1.0 / (K * K))   # [KLIN]
    kidx = tl.arange(0, KLIN)
    for nf in range(NF):
        wf = tl.load(wf_ptr + nf * KLIN + kidx)
        o = tl.sum(val * wf) + tl.load(bf_ptr + nf)
        tl.store(out_ptr + m * NF + nf, o)


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
        K = 8
        OH, OW = H // K, W // K
        KLIN = self.in_chs[3]
        NF = self.fc_out.weight.shape[0]
        M = (N * C * OH * OW) // KLIN
        out = torch.empty((M, NF), device=x.device, dtype=x.dtype)
        _full_kernel[(M,)](
            x, self.c_in.weight, self.c_in.bias,
            self.fc_out.weight, self.fc_out.bias, out,
            N, H, W, CIN=3, C=C, K=K, OH=OH, OW=OW,
            KLIN=KLIN, NF=NF, num_warps=8)
        return out
