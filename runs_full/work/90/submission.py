import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, K,
                BM: tl.constexpr, KP: tl.constexpr,
                N1: tl.constexpr, N1P: tl.constexpr,
                N2: tl.constexpr, N2P: tl.constexpr,
                N3: tl.constexpr, N3P: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, KP)
    offs_n1 = tl.arange(0, N1P)
    offs_n2 = tl.arange(0, N2P)
    offs_n3 = tl.arange(0, N3P)

    m_mask = offs_m < M
    k_mask = offs_k < K

    # load X [BM, KP]
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    # fc1: W1 weight stored [N1, K] contiguous (stride n=K, k=1). Want [KP, N1P].
    w1 = tl.load(w1_ptr + offs_k[:, None] * 1 + offs_n1[None, :] * K,
                 mask=k_mask[:, None] & (offs_n1[None, :] < N1), other=0.0)
    acc1 = tl.dot(x, w1)
    b1 = tl.load(b1_ptr + offs_n1, mask=offs_n1 < N1, other=0.0)
    h1 = tl.maximum(acc1 + b1[None, :], 0.0)

    # fc2: W2 weight stored [N2, N1] contiguous. Want [N1P, N2P].
    w2 = tl.load(w2_ptr + offs_n1[:, None] * 1 + offs_n2[None, :] * N1,
                 mask=(offs_n1[:, None] < N1) & (offs_n2[None, :] < N2), other=0.0)
    acc2 = tl.dot(h1, w2)
    b2 = tl.load(b2_ptr + offs_n2, mask=offs_n2 < N2, other=0.0)
    h2 = tl.maximum(acc2 + b2[None, :], 0.0)

    # fc3: W3 weight stored [N3, N2] contiguous. Want [N2P, N3P].
    w3 = tl.load(w3_ptr + offs_n2[:, None] * 1 + offs_n3[None, :] * N2,
                 mask=(offs_n2[:, None] < N2) & (offs_n3[None, :] < N3), other=0.0)
    acc3 = tl.dot(h2, w3)
    b3 = tl.load(b3_ptr + offs_n3, mask=offs_n3 < N3, other=0.0)
    out = acc3 + b3[None, :]

    # take column 0 (N3 == 1)
    res = tl.sum(tl.where(offs_n3[None, :] == 0, out, 0.0), axis=1)
    tl.store(out_ptr + offs_m, res, mask=m_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class MLPNew(nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        lead = x.shape[:-1]
        K = x.shape[-1]
        xf = x.reshape(-1, K).contiguous()
        M = xf.shape[0]

        out = torch.empty(M, device=x.device, dtype=x.dtype)

        KP = max(16, _next_pow2(K))
        BM = max(16, _next_pow2(M))

        grid = (triton.cdiv(M, BM),)
        _mlp_kernel[grid](
            xf, self.fc_1.weight, self.fc_1.bias,
            self.fc_2.weight, self.fc_2.bias,
            self.fc_3.weight, self.fc_3.bias, out,
            M, K,
            BM=BM, KP=KP,
            N1=80, N1P=128, N2=10, N2P=16, N3=1, N3P=16,
            num_warps=1, num_stages=1,
        )
        return out.reshape(*lead, 1).squeeze(dim=1)
