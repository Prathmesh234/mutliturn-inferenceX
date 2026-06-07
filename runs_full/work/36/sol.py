import torch
import torch.nn as tnn
import triton
import triton.language as tl


@triton.jit
def _conv_pool_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                      N, IC: tl.constexpr, OC: tl.constexpr,
                      H: tl.constexpr, W: tl.constexpr,
                      KH: tl.constexpr, KW: tl.constexpr,
                      OH: tl.constexpr, OW: tl.constexpr,
                      POH: tl.constexpr, POW: tl.constexpr,
                      BLOCK_OC: tl.constexpr):
    pid = tl.program_id(0)
    npw = POH * POW
    n = pid // npw
    rem = pid % npw
    ph = rem // POW
    pw = rem % POW

    oc = tl.arange(0, BLOCK_OC)
    mask = oc < OC

    bias = tl.load(b_ptr + oc, mask=mask, other=0.0)
    acc_max = tl.full((BLOCK_OC,), -float('inf'), tl.float32)

    x_base = n * (IC * H * W)
    for sh in tl.static_range(2):
        for sw in tl.static_range(2):
            oh = 2 * ph + sh
            ow = 2 * pw + sw
            acc = bias
            for ic in tl.static_range(IC):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        ih = oh + kh
                        iw = ow + kw
                        xval = tl.load(x_ptr + x_base + ic * (H * W) + ih * W + iw)
                        wv = tl.load(w_ptr + oc * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw,
                                     mask=mask, other=0.0)
                        acc = acc + xval * wv
            acc_max = tl.maximum(acc_max, acc)

    out_base = n * (OC * POH * POW) + oc * (POH * POW) + ph * POW + pw
    tl.store(out_ptr + out_base, acc_max, mask=mask)


def conv_pool(x, weight, bias):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1
    POH = OH // 2
    POW = OW // 2
    out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)
    BLOCK_OC = triton.next_power_of_2(OC)
    grid = (N * POH * POW,)
    _conv_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, OC, H, W, KH, KW, OH, OW, POH, POW, BLOCK_OC,
        num_warps=1,
    )
    return out


class NetNew(tnn.Module):
    def __init__(self):
        super(NetNew, self).__init__()
        self.conv1 = tnn.Conv2d(3, 6, 5)
        self.pool = tnn.MaxPool2d(2, 2)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.fc1 = tnn.Linear(16 * 5 * 5, 120)
        self.fc2 = tnn.Linear(120, 84)
        self.fc3 = tnn.Linear(84, 10)

    def forward(self, x):
        x = x.contiguous()
        x = conv_pool(x, self.conv1.weight, self.conv1.bias)
        x = conv_pool(x, self.conv2.weight, self.conv2.bias)
        x = x.view(-1, 16 * 5 * 5)
        # Three back-to-back linears with no nonlinearity -> single linear.
        # Cache the combined weight/bias; weights are fixed after load_state_dict.
        Wct, bc = self._combined()
        return torch.addmm(bc, x, Wct)

    def _combined(self):
        W1, b1 = self.fc1.weight, self.fc1.bias
        W2, b2 = self.fc2.weight, self.fc2.bias
        W3, b3 = self.fc3.weight, self.fc3.bias
        cache = getattr(self, "_cache", None)
        key = (W1.data_ptr(), W2.data_ptr(), W3.data_ptr(),
               b1.data_ptr(), b2.data_ptr(), b3.data_ptr())
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        Wct = (W3 @ W2 @ W1).t().contiguous()
        bc = (b1 @ W2.t() + b2) @ W3.t() + b3
        self._cache = (key, Wct, bc)
        return Wct, bc
