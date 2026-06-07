import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, Lup, Lout, OC,
    IC: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.arange(0, BLOCK_OC)[:, None]
    ol = tl.arange(0, BLOCK_L)[None, :]
    oc_mask = oc < OC
    ol_mask = ol < Lout
    m = oc_mask & ol_mask

    acc = tl.zeros((BLOCK_OC, BLOCK_L), dtype=tl.float32)
    for ic in range(IC):
        for k in range(K):
            j = ol + k - PAD
            valid = (j >= 0) & (j < Lup)
            in_idx = (j * L) // Lup
            in_idx = tl.where(valid, in_idx, 0)
            xptr = x_ptr + n * C_in * L + ic * L + in_idx  # [1,BLOCK_L]
            xval = tl.load(xptr, mask=valid & ol_mask, other=0.0)  # [1,BLOCK_L]
            wval = tl.load(w_ptr + oc * IC * K + ic * K + k, mask=oc_mask, other=0.0)  # [BLOCK_OC,1]
            acc += xval * wval

    acc += tl.load(b_ptr + oc, mask=oc_mask, other=0.0)
    out_off = n * OC * Lout + oc * Lout + ol
    tl.store(out_ptr + out_off, acc, mask=m)


class ResizeConv1dNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor,
                 mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=1)

    def forward(self, x):
        x = x.contiguous()
        N, C, L = x.shape
        OC, IC, K = self.conv.weight.shape
        PAD = self.conv.padding[0]
        Lup = int(L * self.scale_factor)
        Lout = Lup + 2 * PAD - K + 1

        out = torch.empty((N, OC, Lout), device=x.device, dtype=x.dtype)
        BLOCK_L = triton.next_power_of_2(Lout)
        BLOCK_OC = triton.next_power_of_2(OC)
        grid = (N,)
        _kernel[grid](
            x, self.conv.weight, self.conv.bias, out,
            N, C, L, Lup, Lout, OC,
            IC=IC, K=K, PAD=PAD, BLOCK_OC=BLOCK_OC, BLOCK_L=BLOCK_L,
            num_warps=1,
        )
        return out
