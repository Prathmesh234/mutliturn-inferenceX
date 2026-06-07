import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def _fused_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                  N, IC: tl.constexpr, OC: tl.constexpr,
                  H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * H * W
    mask = offs < total
    ow = offs % W
    oh = (offs // W) % H
    oc = (offs // (W * H)) % OC
    n = offs // (W * H * OC)

    acc_a = tl.load(b2_ptr + oc, mask=mask, other=0.0)
    acc_b = tl.load(b2_ptr + (oc + OC), mask=mask, other=0.0)

    for ci in range(OC):  # conv2 input channels = conv1 output channels
        for kh2 in range(3):
            hh = oh + kh2 - 1
            vy2 = (hh >= 0) & (hh < H)
            for kw2 in range(3):
                ww = ow + kw2 - 1
                v2 = vy2 & (ww >= 0) & (ww < W)
                # compute conv1_out(n, ci, hh, ww) = max(c1a, c1b)
                c1a = tl.load(b1_ptr + ci) + 0.0 * acc_a
                c1b = tl.load(b1_ptr + (ci + OC)) + 0.0 * acc_a
                for cj in range(IC):
                    for kh1 in range(3):
                        ih = hh + kh1 - 1
                        vy1 = (ih >= 0) & (ih < H)
                        for kw1 in range(3):
                            iw = ww + kw1 - 1
                            valid = mask & v2 & vy1 & (iw >= 0) & (iw < W)
                            xoff = ((n * IC + cj) * H + ih) * W + iw
                            xoff = tl.where(valid, xoff, 0)
                            xv = tl.load(x_ptr + xoff, mask=valid, other=0.0)
                            w1a = tl.load(w1_ptr + (((ci * IC + cj) * 3 + kh1) * 3 + kw1))
                            w1b = tl.load(w1_ptr + ((((ci + OC) * IC + cj) * 3 + kh1) * 3 + kw1))
                            c1a += xv * w1a
                            c1b += xv * w1b
                c1 = tl.maximum(c1a, c1b)
                c1 = tl.where(v2, c1, 0.0)
                w2a = tl.load(w2_ptr + (((oc * OC + ci) * 3 + kh2) * 3 + kw2), mask=mask, other=0.0)
                w2b = tl.load(w2_ptr + ((((oc + OC) * OC + ci) * 3 + kh2) * 3 + kw2), mask=mask, other=0.0)
                acc_a += c1 * w2a
                acc_b += c1 * w2b

    res = tl.maximum(acc_a, acc_b)
    r = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, res + r, mask=mask)


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)
    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class resblockNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblockNew, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.conv1.out_channels
        w1 = self.conv1.filter.weight.contiguous(); b1 = self.conv1.filter.bias.contiguous()
        w2 = self.conv2.filter.weight.contiguous(); b2 = self.conv2.filter.bias.contiguous()
        out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
        total = N * OC * H * W
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _fused_kernel[grid](x, w1, b1, w2, b2, out, N, IC, OC, H, W, BLOCK=BLOCK, num_warps=4)
        return out


if __name__ == "__main__":
    from reference import resblock, get_inputs, get_init_inputs
    a,k=get_init_inputs(); ref=resblock(*a,**k).cuda()
    new=resblockNew(*a,**k).cuda(); new.load_state_dict(ref.state_dict())
    x=get_inputs()[0].cuda()
    import triton.testing as tt
    print('maxerr', (ref(x)-new(x)).abs().max().item())
    t1=tt.do_bench(lambda: ref(x)); t2=tt.do_bench(lambda: new(x))
    print('ref ms', t1, 'new ms', t2, 'speedup', t1/t2)
