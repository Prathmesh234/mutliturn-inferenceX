import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hadamard_kernel(
    x1_ptr, x2_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
    M, idim1, idim2, hdim,
    BM: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr, H: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mask_m = rm < M

    i1 = tl.arange(0, I1)
    i2 = tl.arange(0, I2)
    h = tl.arange(0, H)

    m1 = i1 < idim1
    m2 = i2 < idim2
    mh = h < hdim

    # load x1 [BM, I1], x2 [BM, I2]
    x1 = tl.load(x1_ptr + rm[:, None] * idim1 + i1[None, :],
                 mask=mask_m[:, None] & m1[None, :], other=0.0)
    x2 = tl.load(x2_ptr + rm[:, None] * idim2 + i2[None, :],
                 mask=mask_m[:, None] & m2[None, :], other=0.0)

    # W1 [H, I1], W2 [H, I2], W3 [H, H]
    w1 = tl.load(w1_ptr + h[:, None] * idim1 + i1[None, :],
                 mask=mh[:, None] & m1[None, :], other=0.0)
    w2 = tl.load(w2_ptr + h[:, None] * idim2 + i2[None, :],
                 mask=mh[:, None] & m2[None, :], other=0.0)
    w3 = tl.load(w3_ptr + h[:, None] * hdim + h[None, :],
                 mask=mh[:, None] & mh[None, :], other=0.0)
    b1 = tl.load(b1_ptr + h, mask=mh, other=0.0)
    b2 = tl.load(b2_ptr + h, mask=mh, other=0.0)
    b3 = tl.load(b3_ptr + h, mask=mh, other=0.0)

    # y1[bm,h] = sum_i x1[bm,i]*w1[h,i]
    y1 = tl.sum(x1[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
    y1 = tl.maximum(y1, 0.0)
    y2 = tl.sum(x2[:, None, :] * w2[None, :, :], axis=2) + b2[None, :]
    y2 = tl.maximum(y2, 0.0)
    hm = y1 * y2  # [BM, H]

    # out[bm,o] = sum_h hm[bm,h]*w3[o,h]
    out = tl.sum(hm[:, None, :] * w3[None, :, :], axis=2) + b3[None, :]
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + rm[:, None] * hdim + h[None, :],
             out, mask=mask_m[:, None] & mh[None, :])


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


class HadamardProductNew(nn.Module):
    def __init__(self, idim_1, idim_2, hdim):
        super(HadamardProductNew, self).__init__()
        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, x1, x2):
        idim1 = self.fc_1.in_features
        idim2 = self.fc_2.in_features
        hdim = self.fc_3.in_features

        shape = x1.shape
        x1f = x1.reshape(-1, idim1).contiguous()
        x2f = x2.reshape(-1, idim2).contiguous()
        M = x1f.shape[0]
        out = torch.empty((M, hdim), device=x1.device, dtype=x1.dtype)

        BM = 32
        I1 = _next_pow2(idim1)
        I2 = _next_pow2(idim2)
        H = _next_pow2(hdim)
        grid = (triton.cdiv(M, BM),)
        _hadamard_kernel[grid](
            x1f, x2f,
            self.fc_1.weight, self.fc_1.bias,
            self.fc_2.weight, self.fc_2.bias,
            self.fc_3.weight, self.fc_3.bias,
            out,
            M, idim1, idim2, hdim,
            BM=BM, I1=I1, I2=I2, H=H, num_warps=2,
        )
        return out.reshape(*shape[:-1], hdim)
