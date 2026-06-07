import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW, OC, OHc, OWc, OHp, OWp,
    PAD, KS: tl.constexpr, BLOCK: tl.constexpr):
    pid_no = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_no // OC
    oc = pid_no % OC

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    npool = OHp * OWp
    mask_s = offs < npool
    ph = offs // OWp
    pw = offs % OWp

    acc00 = tl.zeros((BLOCK,), tl.float32)
    acc01 = tl.zeros((BLOCK,), tl.float32)
    acc10 = tl.zeros((BLOCK,), tl.float32)
    acc11 = tl.zeros((BLOCK,), tl.float32)

    oh0 = 2 * ph
    ow0 = 2 * pw

    for ic in range(IC):
        x_base = (n * IC + ic) * IH * IW
        w_base = (oc * IC + ic) * KS * KS
        for kh in range(KS):
            for kw in range(KS):
                w = tl.load(w_ptr + w_base + kh * KS + kw)
                # sub (0,0)
                ih = oh0 - PAD + kh
                iw = ow0 - PAD + kw
                v = mask_s & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                inp = tl.load(x_ptr + x_base + ih * IW + iw, mask=v, other=0.0)
                acc00 += w * inp
                # sub (0,1)
                iw1 = ow0 + 1 - PAD + kw
                v = mask_s & (ih >= 0) & (ih < IH) & (iw1 >= 0) & (iw1 < IW)
                inp = tl.load(x_ptr + x_base + ih * IW + iw1, mask=v, other=0.0)
                acc01 += w * inp
                # sub (1,0)
                ih1 = oh0 + 1 - PAD + kh
                v = mask_s & (ih1 >= 0) & (ih1 < IH) & (iw >= 0) & (iw < IW)
                inp = tl.load(x_ptr + x_base + ih1 * IW + iw, mask=v, other=0.0)
                acc10 += w * inp
                # sub (1,1)
                v = mask_s & (ih1 >= 0) & (ih1 < IH) & (iw1 >= 0) & (iw1 < IW)
                inp = tl.load(x_ptr + x_base + ih1 * IW + iw1, mask=v, other=0.0)
                acc11 += w * inp

    bias = tl.load(b_ptr + oc)
    m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11)) + bias
    m = tl.maximum(m, 0.0)

    out_base = (n * OC + oc) * OHp * OWp
    tl.store(out_ptr + out_base + ph * OWp + pw, m, mask=mask_s)


@triton.jit
def fc_logsoftmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, K, C,
    BN: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    offs_n = tl.arange(0, BN)
    offs_c = tl.arange(0, BC)
    acc = tl.zeros((BN, BC), tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        kmask = offs_k < K
        a = tl.load(x_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_c[None, :] * K + offs_k[:, None],
                    mask=(offs_c[None, :] < C) & kmask[:, None], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    bias = tl.load(b_ptr + offs_c, mask=offs_c < C, other=0.0)
    acc += bias[None, :]
    cmask = offs_c[None, :] < C
    acc = tl.where(cmask, acc, -float('inf'))
    mx = tl.max(acc, axis=1)
    acc = acc - mx[:, None]
    e = tl.exp(acc)
    s = tl.sum(e, axis=1)
    logsm = acc - tl.log(s)[:, None]
    nmask = offs_n[:, None] < N
    tl.store(out_ptr + offs_n[:, None] * C + offs_c[None, :], logsm, mask=nmask & cmask)


def _conv_pool_relu(x, weight, bias, pad):
    N, IC, IH, IW = x.shape
    OC, _, KS, _ = weight.shape
    OHc = IH + 2 * pad - KS + 1
    OWc = IW + 2 * pad - KS + 1
    OHp = OHc // 2
    OWp = OWc // 2
    out = torch.empty((N, OC, OHp, OWp), device=x.device, dtype=x.dtype)
    BLOCK = 64
    grid = (N * OC, triton.cdiv(OHp * OWp, BLOCK))
    conv_pool_relu_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW, OC, OHc, OWc, OHp, OWp,
        pad, KS=KS, BLOCK=BLOCK, num_warps=4, num_stages=2)
    return out


class NetNew(nn.Module):
    def __init__(self):
        super(NetNew, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3 * 3 * 64, 10)

    def forward(self, x):
        x = x.contiguous()
        x = _conv_pool_relu(x, self.conv1.weight, self.conv1.bias, 2)
        x = _conv_pool_relu(x, self.conv2.weight, self.conv2.bias, 1)
        x = _conv_pool_relu(x, self.conv3.weight, self.conv3.bias, 2)
        N = x.shape[0]
        x = x.reshape(N, -1).contiguous()
        K = x.shape[1]
        C = self.fc1.weight.shape[0]
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)
        fc_logsoftmax_kernel[(1,)](
            x, self.fc1.weight, self.fc1.bias, out,
            N, K, C,
            BN=16, BC=16, BK=128, num_warps=2)
        return out
