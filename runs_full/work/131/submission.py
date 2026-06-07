import torch
import torch.nn as nn
import triton
import triton.language as tl


class Conv(nn.Module):

    def __init__(self, conv, in_channels, out_channels):
        super().__init__()
        self.conv_type = conv
        self.relu = nn.ReLU(inplace=True)
        if self.conv_type == 'conv2d':
            self.conv2d = nn.Conv3d(in_channels, out_channels, stride=1,
                kernel_size=(3, 3, 1), padding=(1, 1, 0))
            self.bn2d = nn.InstanceNorm3d(out_channels, affine=True)
        elif self.conv_type == 'conv3d':
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=
                3, stride=1, padding=1)
            self.bn3d = nn.InstanceNorm3d(out_channels, affine=True)
        elif self.conv_type == 'convp3d':
            self.convp3d1 = nn.Conv3d(in_channels, out_channels, stride=1,
                kernel_size=(3, 3, 1), padding=(1, 1, 0))
            self.p3dbn1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.convp3d2 = nn.Conv3d(out_channels, out_channels, stride=1,
                kernel_size=(1, 1, 3), padding=(0, 0, 1))
            self.p3dbn2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        if self.conv_type == 'conv2d':
            x = self.conv2d(x); x = self.bn2d(x); x = self.relu(x)
        elif self.conv_type == 'conv3d':
            x = self.conv3d(x); x = self.bn3d(x); x = self.relu(x)
        elif self.conv_type == 'convp3d':
            x = self.convp3d1(x); x = self.p3dbn1(x)
            x = self.convp3d2(x); x = self.p3dbn2(x); x = self.relu(x)
        return x


@triton.jit
def _conv_in_relu_kernel(
    x_ptr, w_ptr, b_ptr, g_ptr, beta_ptr, out_ptr,
    S, eps,
    Ci: tl.constexpr, Co: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // Co
    co = pid % Co
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    x_base = n * Ci * S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for ci in tl.static_range(Ci):
        w = tl.load(w_ptr + co * Ci + ci).to(tl.float32)
        xv = tl.load(x_ptr + x_base + ci * S + offs, mask=mask, other=0.0).to(tl.float32)
        acc += w * xv
    acc += tl.load(b_ptr + co).to(tl.float32)
    summ = tl.sum(tl.where(mask, acc, 0.0))
    mean = summ / S
    diff = tl.where(mask, acc - mean, 0.0)
    var = tl.sum(diff * diff) / S
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)
    out = (acc - mean) * rstd * g + beta
    out = tl.maximum(out, 0.0)
    o_base = n * Co * S
    tl.store(out_ptr + o_base + co * S + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_cell_kernel(
    x_ptr, w1_ptr, b1_ptr, g1_ptr, beta1_ptr,
    wf_ptr, bf_ptr, gf_ptr, betaf_ptr, out_ptr,
    S, eps,
    C: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # one program per instance n; C channels, S spatial held in registers
    n = tl.program_id(0)
    rows = tl.arange(0, BLOCK_C)              # (C,)
    cols = tl.arange(0, BLOCK_S)             # (S,)
    rmask = rows < C
    cmask = cols < S
    base = n * C * S
    # load X tile (C, S)
    xptrs = base + rows[:, None] * S + cols[None, :]
    X = tl.load(x_ptr + xptrs, mask=(rmask[:, None] & cmask[None, :]), other=0.0).to(tl.float32)

    # ---- stage 1: conv_i1 (C->C) ----
    A = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)
    for ci in tl.static_range(C):
        wcol = tl.load(w1_ptr + rows * C + ci, mask=rmask, other=0.0).to(tl.float32)  # (C,)
        xrow = tl.sum(tl.where(rows[:, None] == ci, X, 0.0), axis=0)                  # (S,)
        A += wcol[:, None] * xrow[None, :]
    b1 = tl.load(b1_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    A += b1[:, None]
    # instance norm over S, per row
    Am = tl.where(cmask[None, :], A, 0.0)
    mean = tl.sum(Am, axis=1) / S                                                     # (C,)
    diff = tl.where(cmask[None, :], A - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / S
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(g1_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    bt1 = tl.load(beta1_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    R = (A - mean[:, None]) * rstd[:, None] * g1[:, None] + bt1[:, None]
    R = tl.maximum(R, 0.0)

    # ---- stage 2: conv_f (C->C) ----
    B = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)
    for ci in tl.static_range(C):
        wcol = tl.load(wf_ptr + rows * C + ci, mask=rmask, other=0.0).to(tl.float32)
        rrow = tl.sum(tl.where(rows[:, None] == ci, R, 0.0), axis=0)
        B += wcol[:, None] * rrow[None, :]
    bf = tl.load(bf_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    B += bf[:, None]
    Bm = tl.where(cmask[None, :], B, 0.0)
    mean2 = tl.sum(Bm, axis=1) / S
    diff2 = tl.where(cmask[None, :], B - mean2[:, None], 0.0)
    var2 = tl.sum(diff2 * diff2, axis=1) / S
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    gf = tl.load(gf_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    btf = tl.load(betaf_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    O = (B - mean2[:, None]) * rstd2[:, None] * gf[:, None] + btf[:, None]
    O = tl.maximum(O, 0.0)
    tl.store(out_ptr + xptrs, O.to(out_ptr.dtype.element_ty),
             mask=(rmask[:, None] & cmask[None, :]))


class CellNew(nn.Module):

    def __init__(self, conv, in_channels, out_channels, double=False):
        super().__init__()
        self.conv_type = conv
        self.double = double
        self.conv_i1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bni1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv(self.conv_type, in_channels, out_channels)
        if self.double:
            self.conv_i2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
            self.bni2 = nn.InstanceNorm3d(in_channels, affine=True)
            self.conv2 = Conv(self.conv_type, in_channels, out_channels)
        self.conv_f = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bnf = nn.InstanceNorm3d(out_channels, affine=True)

    @staticmethod
    def _shape(x):
        if x.dim() == 5:
            N, C = x.shape[0], x.shape[1]
        else:
            N, C = 1, x.shape[0]
        S = x.numel() // (N * C)
        return N, C, S

    def _run(self, xf, N, S, conv, norm):
        Ci = conv.in_channels
        Co = conv.out_channels
        out = torch.empty((N, Co, S), device=xf.device, dtype=xf.dtype)
        w = conv.weight.reshape(Co, Ci).contiguous()
        BLOCK_S = triton.next_power_of_2(S)
        grid = (N * Co,)
        _conv_in_relu_kernel[grid](
            xf, w, conv.bias, norm.weight, norm.bias, out,
            S, norm.eps, Ci, Co, BLOCK_S, num_warps=4)
        return out

    def _fused(self, xf, N, C, S):
        out = torch.empty((N, C, S), device=xf.device, dtype=xf.dtype)
        w1 = self.conv_i1.weight.reshape(C, C).contiguous()
        wf = self.conv_f.weight.reshape(C, C).contiguous()
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_S = triton.next_power_of_2(S)
        _fused_cell_kernel[(N,)](
            xf, w1, self.conv_i1.bias, self.bni1.weight, self.bni1.bias,
            wf, self.conv_f.bias, self.bnf.weight, self.bnf.bias, out,
            S, self.bni1.eps, C, BLOCK_C, BLOCK_S, num_warps=4)
        return out

    def forward(self, x, y=None):
        orig = x.shape
        N, C, S = self._shape(x)
        xf = x.contiguous().reshape(N, C, S)
        Cout = self.conv_f.out_channels
        # fast fused path: identity conv1 (conv_type not a real conv) and no double
        can_fuse = (not self.double) and (self.conv_type not in
                    ('conv2d', 'conv3d', 'convp3d')) and (C == Cout)
        if can_fuse:
            x = self._fused(xf, N, C, S)
        else:
            x = self._run(xf, N, S, self.conv_i1, self.bni1)
            x = self.conv1(x)
            if self.double:
                Ny, Cy, Sy = self._shape(y)
                yf = y.contiguous().reshape(Ny, Cy, Sy)
                y = self._run(yf, Ny, Sy, self.conv_i2, self.bni2)
                y = self.conv2(y)
                x = x + y
            x = self._run(x, N, S, self.conv_f, self.bnf)
        if len(orig) == 5:
            new_shape = (orig[0], Cout) + tuple(orig[2:])
        else:
            new_shape = (Cout,) + tuple(orig[1:])
        return x.reshape(new_shape)
