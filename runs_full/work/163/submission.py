import math
import torch
import numpy as np
from torch import nn
import triton
import triton.language as tl


class MaybeBatchNorm2d(nn.Module):

    def __init__(self, n_ftr, affine, use_bn):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


@triton.jit
def _prebn_kernel(x_ptr, w1_ptr, w2_ptr, ws_ptr, bs_ptr, out_ptr,
                  M, C_in, C_out,
                  BM: tl.constexpr, BCIN: tl.constexpr, BCOUT: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_in = tl.arange(0, BCIN)
    offs_out = tl.arange(0, BCOUT)
    mask_in = offs_in < C_in
    mask_out = offs_out < C_out

    # x block [BM, BCIN]
    x = tl.load(x_ptr + offs_m[:, None] * C_in + offs_in[None, :],
                mask=mask_m[:, None] & mask_in[None, :], other=0.0)

    # w1 transposed -> [BCIN, BCOUT], w1t[i,o] = w1[o,i]  (w1 layout [C_out, C_in])
    w1t = tl.load(w1_ptr + offs_out[None, :] * C_in + offs_in[:, None],
                  mask=mask_in[:, None] & mask_out[None, :], other=0.0)
    h1 = tl.sum(x[:, :, None] * w1t[None, :, :], axis=1)  # [BM, BCOUT]
    h1 = tl.maximum(h1, 0.0)

    # w2 transposed -> [BCOUT, BCOUT], w2t[i,o] = w2[o,i] (w2 layout [C_out, C_out])
    w2t = tl.load(w2_ptr + offs_out[None, :] * C_out + offs_out[:, None],
                  mask=mask_out[:, None] & mask_out[None, :], other=0.0)
    h2 = tl.sum(h1[:, :, None] * w2t[None, :, :], axis=1)  # [BM, BCOUT]

    # shortcut wst -> [BCIN, BCOUT], wst[i,o] = ws[o,i] (ws layout [C_out, C_in])
    wst = tl.load(ws_ptr + offs_out[None, :] * C_in + offs_in[:, None],
                  mask=mask_in[:, None] & mask_out[None, :], other=0.0)
    sc = tl.sum(x[:, :, None] * wst[None, :, :], axis=1)  # [BM, BCOUT]
    bs = tl.load(bs_ptr + offs_out, mask=mask_out, other=0.0)
    sc = sc + bs[None, :]

    pre = h2 + sc
    tl.store(out_ptr + offs_m[:, None] * C_out + offs_out[None, :], pre,
             mask=mask_m[:, None] & mask_out[None, :])


@triton.jit
def _bn_kernel(pre_ptr, out_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr,
               M, C_out, eps, USE_RUNNING: tl.constexpr, BM: tl.constexpr):
    o = tl.program_id(0)
    if o >= C_out:
        return
    if USE_RUNNING:
        mean = tl.load(mean_ptr + o)
        var = tl.load(var_ptr + o)
    else:
        s = 0.0
        ss = 0.0
        for start in range(0, M, BM):
            offs = start + tl.arange(0, BM)
            mask = offs < M
            v = tl.load(pre_ptr + offs * C_out + o, mask=mask, other=0.0)
            s += tl.sum(v)
            ss += tl.sum(v * v)
        Mf = M.to(tl.float32)
        mean = s / Mf
        var = ss / Mf - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + o)
    b = tl.load(beta_ptr + o)
    for start in range(0, M, BM):
        offs = start + tl.arange(0, BM)
        mask = offs < M
        v = tl.load(pre_ptr + offs * C_out + o, mask=mask, other=0.0)
        y = (v - mean) * rstd * g + b
        tl.store(out_ptr + offs * C_out + o, y, mask=mask)


@triton.jit
def _fused_eval_kernel(x_ptr, w1_ptr, w2_ptr, ws_ptr, bs_ptr,
                       gamma_ptr, beta_ptr, mean_ptr, var_ptr, out_ptr,
                       M, HW, C_in, C_out, eps,
                       BM: tl.constexpr, BCIN: tl.constexpr, BCOUT: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BM + tl.arange(0, BM)
    mask_m = p < M
    n = p // HW
    s = p % HW
    offs_in = tl.arange(0, BCIN)
    offs_out = tl.arange(0, BCOUT)
    mask_in = offs_in < C_in
    mask_out = offs_out < C_out

    base_in = n * (C_in * HW) + s
    x = tl.load(x_ptr + base_in[:, None] + offs_in[None, :] * HW,
                mask=mask_m[:, None] & mask_in[None, :], other=0.0)

    w1t = tl.load(w1_ptr + offs_out[None, :] * C_in + offs_in[:, None],
                  mask=mask_in[:, None] & mask_out[None, :], other=0.0)
    h1 = tl.maximum(tl.sum(x[:, :, None] * w1t[None, :, :], axis=1), 0.0)

    w2t = tl.load(w2_ptr + offs_out[None, :] * C_out + offs_out[:, None],
                  mask=mask_out[:, None] & mask_out[None, :], other=0.0)
    h2 = tl.sum(h1[:, :, None] * w2t[None, :, :], axis=1)

    wst = tl.load(ws_ptr + offs_out[None, :] * C_in + offs_in[:, None],
                  mask=mask_in[:, None] & mask_out[None, :], other=0.0)
    sc = tl.sum(x[:, :, None] * wst[None, :, :], axis=1)
    bs = tl.load(bs_ptr + offs_out, mask=mask_out, other=0.0)
    pre = h2 + sc + bs[None, :]

    g = tl.load(gamma_ptr + offs_out, mask=mask_out, other=0.0)
    b = tl.load(beta_ptr + offs_out, mask=mask_out, other=0.0)
    rm = tl.load(mean_ptr + offs_out, mask=mask_out, other=0.0)
    rv = tl.load(var_ptr + offs_out, mask=mask_out, other=1.0)
    y = (pre - rm[None, :]) / tl.sqrt(rv[None, :] + eps) * g[None, :] + b[None, :]

    base_out = n * (C_out * HW) + s
    tl.store(out_ptr + base_out[:, None] + offs_out[None, :] * HW, y,
             mask=mask_m[:, None] & mask_out[None, :])


class FakeRKHSConvNetNew(nn.Module):

    def __init__(self, n_input, n_output, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = MaybeBatchNorm2d(n_output, True, use_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1, stride=
            1, padding=0, bias=True)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.bool_)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.0)

    def init_weights(self, init_scale=1.0):
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        nn.init.constant_(self.conv2.weight, 0.0)

    def forward(self, x):
        assert not self.bn1.use_bn, "use_bn=True not supported in Triton path"
        N, C_in, H, W = x.shape
        C_out = self.conv1.weight.shape[0]
        HW = H * W
        M = N * HW

        bn = self.bn_out.bn
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        w1 = self.conv1.weight.view(C_out, C_in)
        w2 = self.conv2.weight.view(C_out, C_out)
        ws = self.shortcut.weight.view(C_out, C_in)
        bs = self.shortcut.bias
        BCIN = triton.next_power_of_2(C_in)
        BCOUT = triton.next_power_of_2(C_out)

        if not self.training:
            x = x.contiguous()
            out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
            BM = 64
            grid = (triton.cdiv(M, BM),)
            _fused_eval_kernel[grid](x, w1, w2, ws, bs, gamma, beta,
                                     bn.running_mean, bn.running_var, out,
                                     M, HW, C_in, C_out, eps,
                                     BM=BM, BCIN=BCIN, BCOUT=BCOUT, num_warps=4)
            return out

        # training path: batch stats
        xp = x.permute(0, 2, 3, 1).contiguous().view(M, C_in)
        w1 = w1.contiguous(); w2 = w2.contiguous(); ws = ws.contiguous()
        bs = bs.contiguous()
        pre = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
        BM = 128
        grid = (triton.cdiv(M, BM),)
        _prebn_kernel[grid](xp, w1, w2, ws, bs, pre, M, C_in, C_out,
                            BM=BM, BCIN=BCIN, BCOUT=BCOUT, num_warps=4)
        out = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
        _bn_kernel[(C_out,)](pre, out, gamma.contiguous(), beta.contiguous(),
                             bn.running_mean, bn.running_var, M, C_out, eps,
                             USE_RUNNING=False, BM=256, num_warps=4)
        out = out.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()
        return out
