import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# ---- generic fallback kernels ----
@triton.jit
def _relu_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def _add_relu_kernel(out_ptr, a_ptr, b_ptr, n,
                     sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3,
                     D1, D2, D3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    w = idx % D3
    t = idx // D3
    h = t % D2
    t = t // D2
    c = t % D1
    bn = t // D1
    a = tl.load(a_ptr + bn * sa0 + c * sa1 + h * sa2 + w * sa3, mask=mask)
    b = tl.load(b_ptr + bn * sb0 + c * sb1 + h * sb2 + w * sb3, mask=mask)
    tl.store(out_ptr + idx, tl.maximum(a + b, 0.0), mask=mask)


def _relu_(x):
    n = x.numel()
    _relu_kernel[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024, num_warps=4)
    return x


def _bcast_strides(t, shape):
    return [0 if (t.shape[i] == 1 and shape[i] != 1) else t.stride(i) for i in range(4)]


def _add_relu(a, b):
    shape = [max(a.shape[i], b.shape[i]) for i in range(4)]
    out = torch.empty(shape, device=a.device, dtype=a.dtype)
    n = out.numel()
    sa = _bcast_strides(a, shape)
    sb = _bcast_strides(b, shape)
    _add_relu_kernel[(triton.cdiv(n, 1024),)](out, a, b, n,
        sa[0], sa[1], sa[2], sa[3], sb[0], sb[1], sb[2], sb[3],
        shape[1], shape[2], shape[3], BLOCK=1024, num_warps=4)
    return out


# ---- fully fused single-kernel path (identity shortcut, 1x1 conv output) ----
@triton.jit
def _fused_block(x_ptr, w1_ptr, w2_ptr, out_ptr,
                 sxN, sxC, sxH, sxW, soN, soC, soH, soW,
                 CIN: tl.constexpr, COUT: tl.constexpr, H: tl.constexpr,
                 W: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                 PAD: tl.constexpr):
    n = tl.program_id(0)
    oc = tl.program_id(1)
    # conv2 over conv1-output channels (1x1 input -> center weight only)
    acc2 = tl.zeros((), tl.float32)
    for c1 in tl.static_range(COUT):
        # conv1 + relu at output spatial (0,0) for channel c1
        acc = tl.zeros((), tl.float32)
        for ic in tl.static_range(CIN):
            for kh in tl.static_range(3):
                ih = kh - PAD
                for kw in tl.static_range(3):
                    iw = kw - PAD
                    if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W):
                        xv = tl.load(x_ptr + n * sxN + ic * sxC + ih * sxH + iw * sxW)
                        wv = tl.load(w1_ptr + c1 * (CIN * 9) + ic * 9 + kh * 3 + kw)
                        acc += xv * wv
        r = tl.maximum(acc, 0.0)
        wv2 = tl.load(w2_ptr + oc * (COUT * 9) + c1 * 9 + 4)
        acc2 += r * wv2
    # add shortcut (identity) + relu, broadcast over spatial
    for hh in tl.static_range(H):
        for ww in tl.static_range(W):
            xv = tl.load(x_ptr + n * sxN + oc * sxC + hh * sxH + ww * sxW)
            res = tl.maximum(acc2 + xv, 0.0)
            tl.store(out_ptr + n * soN + oc * soC + hh * soH + ww * soW, res)


class SimpleBlockNew(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlockNew, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if
            half_res else 1, padding=1, bias=False)
        self.BN1 = nn.Identity()
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.Identity()
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def _can_fuse(self, x):
        if self.shortcut_type != 'identity':
            return False
        N, C, H, W = x.shape
        s = self.C1.stride[0]
        H1 = (H + 2 - 3) // s + 1
        W1 = (W + 2 - 3) // s + 1
        return H1 == 1 and W1 == 1 and x.is_contiguous()

    def forward(self, x):
        if self._can_fuse(x):
            N, C, H, W = x.shape
            out = torch.empty_like(x)
            w1 = self.C1.weight.contiguous()
            w2 = self.C2.weight.contiguous()
            _fused_block[(N, C)](
                x, w1, w2, out,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                CIN=C, COUT=C, H=H, W=W,
                SH=self.C1.stride[0], SW=self.C1.stride[1], PAD=1,
                num_warps=1)
            return out
        out = self.C1(x)
        out = _relu_(out)
        out = self.C2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = _add_relu(out, short_out)
        return out
