import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr,
                w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                w3_ptr, b3_ptr, w4_ptr, b4_ptr,
                out_ptr,
                M,
                N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr, N4: tl.constexpr,
                K1: tl.constexpr,
                BLOCK_M: tl.constexpr, BK1: tl.constexpr,
                BN1: tl.constexpr, BN2: tl.constexpr, BN3: tl.constexpr, BN4: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # load input tile [BLOCK_M, BK1], padded over K
    offs_k = tl.arange(0, BK1)
    x = tl.load(x_ptr + offs_m[:, None] * K1 + offs_k[None, :],
                mask=m_mask[:, None] & (offs_k[None, :] < K1), other=0.0)

    # layer 1: K1 -> N1
    on = tl.arange(0, BN1)
    w = tl.load(w1_ptr + on[:, None] * K1 + offs_k[None, :],
                mask=(on[:, None] < N1) & (offs_k[None, :] < K1), other=0.0)
    acc = tl.dot(x, tl.trans(w))
    b = tl.load(b1_ptr + on, mask=on < N1, other=0.0)
    h1 = tl.maximum(acc + b[None, :], 0.0)  # [BM, BN1]

    # layer 2: N1 -> N2
    ok = tl.arange(0, BN1)
    on = tl.arange(0, BN2)
    w = tl.load(w2_ptr + on[:, None] * N1 + ok[None, :], mask=(on[:, None] < N2) & (ok[None, :] < N1), other=0.0)
    acc = tl.dot(h1, tl.trans(w))
    b = tl.load(b2_ptr + on, mask=on < N2, other=0.0)
    h2 = tl.maximum(acc + b[None, :], 0.0)

    # layer 3: N2 -> N3
    ok = tl.arange(0, BN2)
    on = tl.arange(0, BN3)
    w = tl.load(w3_ptr + on[:, None] * N2 + ok[None, :], mask=(on[:, None] < N3) & (ok[None, :] < N2), other=0.0)
    acc = tl.dot(h2, tl.trans(w))
    b = tl.load(b3_ptr + on, mask=on < N3, other=0.0)
    h3 = tl.maximum(acc + b[None, :], 0.0)

    # layer 4: N3 -> N4
    ok = tl.arange(0, BN3)
    on = tl.arange(0, BN4)
    w = tl.load(w4_ptr + on[:, None] * N3 + ok[None, :], mask=(on[:, None] < N4) & (ok[None, :] < N3), other=0.0)
    acc = tl.dot(h3, tl.trans(w))
    b = tl.load(b4_ptr + on, mask=on < N4, other=0.0)
    out = acc + b[None, :]

    on = tl.arange(0, BN4)
    tl.store(out_ptr + offs_m[:, None] * N4 + on[None, :],
             out, mask=m_mask[:, None] & (on[None, :] < N4))


class SimpleMLPNew(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout_probability):
        super(SimpleMLPNew, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dropout_probability = dropout_probability
        self.l1 = nn.Linear(self.n_inputs, self.n_inputs * 2)
        self.l2 = nn.Linear(self.n_inputs * 2, self.n_inputs * 4)
        self.l3 = nn.Linear(self.n_inputs * 4, self.n_inputs * 8)
        self.l4 = nn.Linear(self.n_inputs * 8, self.n_outputs)
        self.dropout = nn.Dropout(self.dropout_probability)

    def forward(self, X):
        orig_shape = X.shape
        K1 = self.n_inputs
        x = X.reshape(-1, K1).contiguous()
        M = x.shape[0]
        N1, N2, N3, N4 = K1 * 2, K1 * 4, K1 * 8, self.n_outputs
        out = torch.empty((M, N4), device=x.device, dtype=x.dtype)
        BLOCK_M = max(16, triton.next_power_of_2(M))
        grid = (1,)
        _mlp_kernel[grid](x,
                          self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias,
                          self.l3.weight, self.l3.bias, self.l4.weight, self.l4.bias,
                          out, M,
                          N1, N2, N3, N4, K1,
                          BLOCK_M, max(16, triton.next_power_of_2(K1)),
                          max(16, triton.next_power_of_2(N1)),
                          max(16, triton.next_power_of_2(N2)),
                          max(16, triton.next_power_of_2(N3)),
                          max(16, triton.next_power_of_2(N4)),
                          num_warps=4)
        return out.reshape(*orig_shape[:-1], N4)
