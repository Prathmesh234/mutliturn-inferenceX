import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l2norm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, C, HW, eps,
                   HAS_BIAS: tl.constexpr, NP: tl.constexpr, CP: tl.constexpr, HWP: tl.constexpr):
    nidx = tl.arange(0, NP)
    cidx = tl.arange(0, CP)
    hidx = tl.arange(0, HWP)
    nmask = nidx < N
    cmask = cidx < C
    hmask = hidx < HW
    off = (nidx[:, None, None] * C + cidx[None, :, None]) * HW + hidx[None, None, :]
    m = nmask[:, None, None] & cmask[None, :, None] & hmask[None, None, :]
    v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    sumsq = tl.sum(v * v, axis=1)
    inv = 1.0 / (tl.sqrt(sumsq) + eps)
    wc = tl.load(w_ptr + cidx, mask=cmask, other=0.0).to(tl.float32)
    scaled = v * inv[:, None, :]
    y = scaled * wc[None, :, None]
    if HAS_BIAS:
        bc = tl.load(b_ptr + cidx, mask=cmask, other=0.0).to(tl.float32)
        y += bc[None, :, None]
    tl.store(out_ptr + off, y, mask=m)


class Scale(nn.Module):
    def __init__(self, nchannels, bias=True, init_scale=1.0):
        super().__init__()
        self.nchannels = nchannels
        self.weight = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale=1.0):
        self.weight.data.fill_(init_scale)
        if self.bias is not None:
            self.bias.data.fill_(0.0)


class L2NormNew(nn.Module):
    def __init__(self, nchannels, bias=True):
        super().__init__()
        self.scale = Scale(nchannels, bias=bias)
        self.nchannels = nchannels
        self.eps = 1e-06

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        HW = H * W
        out = torch.empty_like(x)
        w = self.scale.weight
        b = self.scale.bias
        has_bias = b is not None
        _l2norm_kernel[(1,)](x, w, b if has_bias else x, out, N, C, HW, self.eps,
                             HAS_BIAS=has_bias,
                             NP=triton.next_power_of_2(N),
                             CP=triton.next_power_of_2(C),
                             HWP=triton.next_power_of_2(HW), num_warps=4)
        return out
