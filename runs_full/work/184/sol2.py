import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_net(x_ptr,
               w1_ptr, b1_ptr, w2_ptr, b2_ptr,
               w3_ptr, b3_ptr, wo_ptr, bo_ptr,
               out_ptr,
               M,
               K0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
               N3: tl.constexpr, NO: tl.constexpr,
               BM: tl.constexpr, BK0: tl.constexpr, BN1: tl.constexpr,
               BN2: tl.constexpr, BN3: tl.constexpr, BNO: tl.constexpr):
    offs_m = tl.arange(0, BM)
    m_mask = offs_m < M

    # ---- layer 1: x[M,K0] @ W1[N1,K0]^T -> sigmoid ----
    rk0 = tl.arange(0, BK0)
    x = tl.load(x_ptr + offs_m[:, None] * K0 + rk0[None, :],
                mask=m_mask[:, None] & (rk0[None, :] < K0), other=0.0)
    rn1 = tl.arange(0, BN1)
    w1 = tl.load(w1_ptr + rn1[:, None] * K0 + rk0[None, :],
                 mask=(rn1[:, None] < N1) & (rk0[None, :] < K0), other=0.0)
    acc = tl.dot(x, tl.trans(w1))
    b1 = tl.load(b1_ptr + rn1, mask=rn1 < N1, other=0.0)
    h1 = 1.0 / (1.0 + tl.exp(-(acc + b1[None, :])))  # [BM, BN1]

    # ---- layer 2 ----
    rn2 = tl.arange(0, BN2)
    w2 = tl.load(w2_ptr + rn2[:, None] * N1 + rn1[None, :],
                 mask=(rn2[:, None] < N2) & (rn1[None, :] < N1), other=0.0)
    acc = tl.dot(h1, tl.trans(w2))
    b2 = tl.load(b2_ptr + rn2, mask=rn2 < N2, other=0.0)
    h2 = 1.0 / (1.0 + tl.exp(-(acc + b2[None, :])))  # [BM, BN2]

    # ---- layer 3: relu ----
    rn3 = tl.arange(0, BN3)
    w3 = tl.load(w3_ptr + rn3[:, None] * N2 + rn2[None, :],
                 mask=(rn3[:, None] < N3) & (rn2[None, :] < N2), other=0.0)
    acc = tl.dot(h2, tl.trans(w3))
    b3 = tl.load(b3_ptr + rn3, mask=rn3 < N3, other=0.0)
    h3 = tl.maximum(acc + b3[None, :], 0.0)  # [BM, BN3]

    # ---- output layer: sigmoid ----
    rno = tl.arange(0, BNO)
    wo = tl.load(wo_ptr + rno[:, None] * N3 + rn3[None, :],
                 mask=(rno[:, None] < NO) & (rn3[None, :] < N3), other=0.0)
    acc = tl.dot(h3, tl.trans(wo))
    bo = tl.load(bo_ptr + rno, mask=rno < NO, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-(acc + bo[None, :])))  # [BM, BNO]

    out_col = tl.sum(tl.where(rno[None, :] == 0, out, 0.0), axis=1)  # [BM]
    tl.store(out_ptr + offs_m, out_col, mask=m_mask)


class Neural_NetNew(nn.Module):
    def __init__(self, D_in):
        super(Neural_NetNew, self).__init__()
        self.fc1 = nn.Linear(D_in, 100)
        self.relu1 = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(50, 20)
        self.relu3 = nn.ReLU()
        self.fc_output = nn.Linear(20, 1)
        self.fc_output_activation = nn.Sigmoid()

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2d = x.reshape(-1, K).contiguous().to(torch.float32)
        M = x2d.shape[0]
        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        BM = max(16, triton.next_power_of_2(M))
        _fused_net[(1,)](
            x2d,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias,
            self.fc_output.weight, self.fc_output.bias,
            out, M,
            K0=K, N1=100, N2=50, N3=20, NO=1,
            BM=BM, BK0=max(16, triton.next_power_of_2(K)),
            BN1=128, BN2=64, BN3=32, BNO=16,
            num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], 1)
