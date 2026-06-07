import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _crelu_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_out, C, HW, C2,
                  BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    CHW = C * HW
    n = offs // (C2 * HW)
    rem = offs - n * (C2 * HW)
    c_full = rem // HW
    spatial = rem - c_full * HW
    is_hi = c_full >= C
    c_in = c_full - tl.where(is_hi, C, 0)
    sign = tl.where(is_hi, -1.0, 1.0)
    in_idx = n * CHW + c_in * HW + spatial
    x = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    w = tl.load(w_ptr + c_full, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_full, mask=mask, other=0.0)
    val = tl.maximum(sign * x * w + b, 0.0)
    tl.store(out_ptr + offs, val, mask=mask)


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

    def forward(self, x):
        y = x * self.weight
        if self.bias is not None:
            y += self.bias
        return y


class CReLUNew(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(2 * nchannels)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = nchannels
        self.out_channels = 2 * nchannels

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.contiguous()
        C2 = 2 * C
        out = torch.empty((N, C2, H, W), device=x.device, dtype=x.dtype)
        HW = H * W
        n_out = out.numel()
        w = self.scale.weight.reshape(-1).contiguous()
        b = self.scale.bias.reshape(-1).contiguous()
        BLOCK_SIZE = triton.next_power_of_2(n_out)
        _crelu_kernel[(1,)](x, w, b, out, n_out, C, HW, C2,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
