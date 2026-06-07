import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(x_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
                  N, C_out, L_in, L_out,
                  sxn, sxc, sxl, swco, swci, swk, son, soc, sol, srn, src, srl,
                  C_in: tl.constexpr, K: tl.constexpr, pad: tl.constexpr,
                  dilation: tl.constexpr, ACT: tl.constexpr,
                  ADD_RES: tl.constexpr, BLOCK_L: tl.constexpr):
    pid = tl.program_id(0)
    pid_l = tl.program_id(1)
    n = pid // C_out
    co = pid % C_out
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L_out
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    for k in range(K):
        in_l = offs_l - pad + k * dilation
        mask_in = (in_l >= 0) & (in_l < L_in) & mask_l
        for ci in range(C_in):
            w = tl.load(w_ptr + co * swco + ci * swci + k * swk)
            x = tl.load(x_ptr + n * sxn + ci * sxc + in_l * sxl,
                        mask=mask_in, other=0.0)
            acc += w * x
    acc += tl.load(b_ptr + co)
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0
    if ADD_RES:
        r = tl.load(res_ptr + n * srn + co * src + offs_l * srl,
                    mask=mask_l, other=0.0)
        acc += r
    tl.store(out_ptr + n * son + co * soc + offs_l * sol, acc, mask=mask_l)


def _run_conv(x, w, b, res, L_out, pad, dilation, act, add_res):
    N, C_in, L_in = x.shape
    C_out = w.shape[0]
    K = w.shape[2]
    out = torch.empty((N, C_out, L_out), device=x.device, dtype=x.dtype)
    BLOCK_L = triton.next_power_of_2(L_out)
    grid = (N * C_out, triton.cdiv(L_out, BLOCK_L))
    if res is None:
        res = x
        srn = src = srl = 0
    else:
        srn, src, srl = res.stride()
    conv1d_kernel[grid](
        x, w, b, res, out,
        N, C_out, L_in, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        srn, src, srl,
        C_in=C_in, K=K, pad=pad, dilation=dilation, ACT=act,
        ADD_RES=add_res, BLOCK_L=BLOCK_L, num_warps=4)
    return out


class DilatedResConvNew(nn.Module):
    def __init__(self, channels, dilation=1, activation='relu', padding=1,
                 kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels
        self.act_name = activation
        if activation == 'relu':
            self.activation = lambda *a, **k: F.relu(*a, **k, inplace=True)
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'glu':
            self.activation = F.glu
            in_channels = channels // 2
        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels,
                                      kernel_size=kernel_size, stride=1,
                                      padding=dilation * padding,
                                      dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels, kernel_size=1,
                                  bias=True)
        self._dilation = dilation
        self._padding = dilation * padding

    def forward(self, input):
        # GLU path not supported by fused kernel; fall back to reference math.
        if self.act_name == 'glu':
            x = input
            if self.left_pad > 0:
                x = F.pad(x, (self.left_pad, 0))
            x = self.dilated_conv(x)
            x = self.activation(x)
            x = self.conv_1x1(x)
            return input + x

        orig_shape = input.shape
        x = input
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.contiguous()

        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))

        N, C_in, L_in = x.shape
        K = self.dilated_conv.weight.shape[2]
        L_out = L_in + 2 * self._padding - self._dilation * (K - 1)

        act_code = 1 if self.act_name == 'relu' else 2

        h = _run_conv(x, self.dilated_conv.weight, self.dilated_conv.bias,
                      None, L_out, self._padding, self._dilation,
                      act_code, False)

        res = x if self.left_pad == 0 else input
        if res.dim() == 2:
            res = res.unsqueeze(0)
        res = res.contiguous()

        out = _run_conv(h, self.conv_1x1.weight, self.conv_1x1.bias,
                        res, h.shape[2], 0, 1, 0, True)

        if len(orig_shape) == 2:
            out = out.squeeze(0)
        return out
