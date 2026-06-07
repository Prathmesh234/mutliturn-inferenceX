import torch
import triton
import triton.language as tl


@triton.jit
def fused_all(x_ptr, w_ptr, b_ptr, gamma_ptr, beta_ptr, out_ptr,
              N: tl.constexpr, Cin: tl.constexpr, Cout, H: tl.constexpr,
              W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr, eps):
    c = tl.program_id(0)
    HW = H * W
    n = tl.arange(0, N)[:, None, None]
    h = tl.arange(0, H)[None, :, None]
    w = tl.arange(0, W)[None, None, :]
    acc = tl.zeros((N, H, W), tl.float32)
    for cin in range(Cin):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1
                iw = w + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (n >= 0)
                xoff = n * Cin * HW + cin * HW + ih * W + iw
                xv = tl.load(x_ptr + xoff, mask=valid, other=0.0)
                wval = tl.load(w_ptr + c * Cin * 9 + cin * 9 + kh * 3 + kw)
                acc += xv * wval
    conv = acc + tl.load(b_ptr + c)
    cnt = N * H * W
    mean = tl.sum(conv) / cnt
    var = tl.sum(conv * conv) / cnt - mean * mean
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    scale = gamma / tl.sqrt(var + eps)
    shift = beta - mean * scale
    act = tl.maximum(conv * scale + shift, 0.0)
    act5 = tl.reshape(act, (N, OH, 2, OW, 2))
    p = tl.max(tl.max(act5, axis=4), axis=2)
    nn = tl.arange(0, N)[:, None, None]
    ohh = tl.arange(0, OH)[None, :, None]
    oww = tl.arange(0, OW)[None, None, :]
    ooff = nn * Cout * OH * OW + c * OH * OW + ohh * OW + oww
    tl.store(out_ptr + ooff, p)


class ConvBlockNew(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, padding=1)
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=out_channels,
            momentum=1.0, track_running_stats=False)
        self.cached_support_features = None

    def forward(self, x, is_support=False):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        Cout = self.conv2d.out_channels
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous()
        gamma = self.batchnorm2d.weight.contiguous()
        beta = self.batchnorm2d.bias.contiguous()
        eps = self.batchnorm2d.eps
        OH, OW = H // 2, W // 2
        out = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
        fused_all[(Cout,)](x, weight, bias, gamma, beta, out,
                           N, Cin, Cout, H, W, OH, OW, eps, num_warps=2)
        if is_support:
            self.cached_support_features = out.detach()
        return out
