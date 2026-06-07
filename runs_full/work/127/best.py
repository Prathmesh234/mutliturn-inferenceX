import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, N, C, H, W, K, CT, R, S, pad,
                  BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    w_idx = pid % W
    hw = pid // W
    h_idx = hw % H
    n_idx = hw // H

    # copy x into first C channels
    c_off = tl.arange(0, BLOCK_C)
    mask_c = c_off < C
    xv = tl.load(x_ptr + ((n_idx * C + c_off) * H + h_idx) * W + w_idx, mask=mask_c)
    tl.store(out_ptr + ((n_idx * CT + c_off) * H + h_idx) * W + w_idx, xv, mask=mask_c)

    # conv + relu into next K channels
    k_off = tl.arange(0, BLOCK_K)
    mask_k = k_off < K
    acc = tl.zeros((BLOCK_K,), tl.float32)
    for c in range(C):
        for r in range(R):
            for s in range(S):
                ih = h_idx + r - pad
                iw = w_idx + s - pad
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xval = tl.load(x_ptr + ((n_idx * C + c) * H + ih) * W + iw,
                               mask=in_bounds, other=0.0)
                wval = tl.load(w_ptr + ((k_off * C + c) * R + r) * S + s,
                               mask=mask_k, other=0.0)
                acc += xval * wval
    acc = tl.maximum(acc, 0.0)
    out_off = ((n_idx * CT + (C + k_off)) * H + h_idx) * W + w_idx
    tl.store(out_ptr + out_off, acc, mask=mask_k)


class make_denseNew(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_denseNew, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.nChannels = nChannels

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        w = self.conv.weight
        K, _, R, S = w.shape
        pad = self.conv.padding[0]
        CT = C + K
        out = torch.empty((N, CT, H, W), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_K = triton.next_power_of_2(K)
        grid = (N * H * W,)
        _fused_kernel[grid](x, w, out, N, C, H, W, K, CT, R, S, pad,
                            BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, num_warps=1)
        return out
