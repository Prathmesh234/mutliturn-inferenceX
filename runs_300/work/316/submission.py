import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _adv_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, K: tl.constexpr, H: tl.constexpr, slope: tl.constexpr,
                BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_k = tl.arange(0, K)
    offs_h = tl.arange(0, H)

    # load x [BLOCK_M, K]
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None], other=0.0)

    # W1 weight stored [H, K] -> w1[k,n] = weight[n,k]
    w1 = tl.load(w1_ptr + offs_h[None, :] * K + offs_k[:, None])  # [K, H]
    b1 = tl.load(b1_ptr + offs_h)  # [H]
    # h1[m,n] = sum_k x[m,k]*w1[k,n]
    h1 = tl.sum(x[:, :, None] * w1[None, :, :], axis=1) + b1[None, :]  # [BM, H]
    a1 = tl.where(h1 > 0, h1, slope * h1)

    # W2 [H, H]
    w2 = tl.load(w2_ptr + offs_h[None, :] * H + offs_h[:, None])  # [H, H] : w2[j,n]=weight[n,j]
    b2 = tl.load(b2_ptr + offs_h)
    h2 = tl.sum(a1[:, :, None] * w2[None, :, :], axis=1) + b2[None, :]
    a2 = tl.where(h2 > 0, h2, slope * h2)

    # W3 [1, H]
    w3 = tl.load(w3_ptr + offs_h)  # [H]
    b3 = tl.load(b3_ptr + 0)
    out = tl.sum(a2 * w3[None, :], axis=1) + b3  # [BM]
    tl.store(out_ptr + offs_m, out, mask=mask_m)


class AdvNew(nn.Module):
    def __init__(self, dim_inputs, dropout):
        super(AdvNew, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 32)
        self.affine2 = nn.Linear(32, 32)
        self.adv_head = nn.Linear(32, 1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2d = x.reshape(-1, K).contiguous()
        M = x2d.shape[0]
        H = 32
        out = torch.empty((M,), device=x.device, dtype=x.dtype)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _adv_kernel[grid](
            x2d, self.affine1.weight, self.affine1.bias,
            self.affine2.weight, self.affine2.bias,
            self.adv_head.weight, self.adv_head.bias, out,
            M, K, H, 0.01, BLOCK_M=BLOCK_M, num_warps=4)
        return out.reshape(*orig_shape[:-1], 1)
