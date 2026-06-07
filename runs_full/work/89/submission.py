import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x1_ptr, x2_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                  M, K1, K2, Kr, D3r, D4r,
                  BLOCK_M: tl.constexpr, BK: tl.constexpr,
                  BD3: tl.constexpr, BD4: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    off_k = tl.arange(0, BK)
    from_x1 = off_k < K1
    real_k = off_k < Kr
    v1 = tl.load(x1_ptr + offs_m[:, None] * K1 + off_k[None, :],
                 mask=mask_m[:, None] & from_x1[None, :], other=0.0)
    v2 = tl.load(x2_ptr + offs_m[:, None] * K2 + (off_k - K1)[None, :],
                 mask=mask_m[:, None] & (~from_x1[None, :]) & real_k[None, :],
                 other=0.0)
    x = tl.where(from_x1[None, :], v1, v2)
    off_d3 = tl.arange(0, BD3)
    md3 = off_d3 < D3r
    w1 = tl.load(w1_ptr + off_d3[None, :] * Kr + off_k[:, None],
                 mask=md3[None, :] & real_k[:, None], other=0.0)
    h = tl.dot(x, w1)
    b1 = tl.load(b1_ptr + off_d3, mask=md3, other=0.0)
    h = tl.maximum(h + b1[None, :], 0.0)
    off_d4 = tl.arange(0, BD4)
    md4 = off_d4 < D4r
    w2 = tl.load(w2_ptr + off_d4[None, :] * D3r + off_d3[:, None],
                 mask=md4[None, :] & md3[:, None], other=0.0)
    out = tl.dot(h, w2)
    b2 = tl.load(b2_ptr + off_d4, mask=md4, other=0.0)
    out = out + b2[None, :]
    tl.store(out_ptr + offs_m[:, None] * D4r + off_d4[None, :], out,
             mask=mask_m[:, None] & md4[None, :])


class MergeLayerNew(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x1 = x1.contiguous(); x2 = x2.contiguous()
        M, K1 = x1.shape
        K2 = x2.shape[1]
        D3r, Kr = self.fc1.weight.shape
        D4r = self.fc2.weight.shape[0]
        out = torch.empty((M, D4r), device=x1.device, dtype=torch.float32)
        BK = max(16, triton.next_power_of_2(Kr))
        BD3 = max(16, triton.next_power_of_2(D3r))
        BD4 = max(16, triton.next_power_of_2(D4r))
        BM = max(16, triton.next_power_of_2(M))
        grid = (triton.cdiv(M, BM),)
        _fused_kernel[grid](x1, x2, self.fc1.weight, self.fc1.bias,
                            self.fc2.weight, self.fc2.bias, out,
                            M, K1, K2, Kr, D3r, D4r,
                            BLOCK_M=BM, BK=BK, BD3=BD3, BD4=BD4,
                            num_warps=1, num_stages=1)
        return out
