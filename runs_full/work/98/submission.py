import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, K,
                sxm, sxk, som, son,
                N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr,
                BM: tl.constexpr, BK: tl.constexpr,
                BN1: tl.constexpr, BN2: tl.constexpr, BN3: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    ok1 = tl.arange(0, BN1)
    ok2 = tl.arange(0, BN2)
    ok3 = tl.arange(0, BN3)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    w1 = tl.load(w1_ptr + ok1[:, None] * K + offs_k[None, :],
                 mask=(ok1[:, None] < N1) & (offs_k[None, :] < K), other=0.0)
    h1 = tl.dot(x, tl.trans(w1))
    b1 = tl.load(b1_ptr + ok1, mask=ok1 < N1, other=0.0)
    h1 = tl.maximum(h1 + b1[None, :], 0.0)
    w2 = tl.load(w2_ptr + ok2[:, None] * N1 + ok1[None, :],
                 mask=(ok2[:, None] < N2) & (ok1[None, :] < N1), other=0.0)
    h2 = tl.dot(h1, tl.trans(w2))
    b2 = tl.load(b2_ptr + ok2, mask=ok2 < N2, other=0.0)
    h2 = tl.maximum(h2 + b2[None, :], 0.0)
    w3 = tl.load(w3_ptr + ok3[:, None] * N2 + ok2[None, :],
                 mask=(ok3[:, None] < N3) & (ok2[None, :] < N2), other=0.0)
    out = tl.dot(h2, tl.trans(w3))
    b3 = tl.load(b3_ptr + ok3, mask=ok3 < N3, other=0.0)
    out = out + b3[None, :]
    tl.store(out_ptr + offs_m[:, None] * som + ok3[None, :] * son, out,
             mask=(offs_m[:, None] < M) & (ok3[None, :] < N3))


class MLP_multiple_classNew(nn.Module):
    def __init__(self, dim, n_labels, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, n_labels)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)
        self.n_labels = n_labels

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        xf = x.reshape(-1, K)
        if not xf.is_contiguous():
            xf = xf.contiguous()
        M = xf.shape[0]
        N1, N2, N3 = 80, 10, self.n_labels
        out = torch.empty((M, N3), device=xf.device, dtype=torch.float32)
        BM = max(triton.next_power_of_2(M), 16)
        BK = max(triton.next_power_of_2(K), 16)
        BN1 = max(triton.next_power_of_2(N1), 16)
        BN2 = max(triton.next_power_of_2(N2), 16)
        BN3 = max(triton.next_power_of_2(N3), 16)
        grid = (triton.cdiv(M, BM),)
        _mlp_kernel[grid](xf, self.fc_1.weight, self.fc_1.bias,
                          self.fc_2.weight, self.fc_2.bias,
                          self.fc_3.weight, self.fc_3.bias, out,
                          M, K, xf.stride(0), xf.stride(1),
                          out.stride(0), out.stride(1),
                          N1=N1, N2=N2, N3=N3,
                          BM=BM, BK=BK, BN1=BN1, BN2=BN2, BN3=BN3,
                          num_warps=1, num_stages=1)
        return out.reshape(*orig_shape[:-1], N3)
