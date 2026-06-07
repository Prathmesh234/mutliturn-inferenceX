import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bpnet_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                  M, K0, L1, L2, N,
                  sxm, sxk,
                  som, son,
                  BLOCK_M: tl.constexpr,
                  PK0: tl.constexpr, PL1: tl.constexpr, PL2: tl.constexpr, PN: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk0 = tl.arange(0, PK0)
    rl1 = tl.arange(0, PL1)
    rl2 = tl.arange(0, PL2)
    rn = tl.arange(0, PN)

    x = tl.load(x_ptr + offs_m[:, None] * sxm + rk0[None, :] * sxk,
                mask=(offs_m[:, None] < M) & (rk0[None, :] < K0), other=0.0)

    w1 = tl.load(w1_ptr + rl1[:, None] * K0 + rk0[None, :],
                 mask=(rl1[:, None] < L1) & (rk0[None, :] < K0), other=0.0)
    h1 = tl.dot(x, tl.trans(w1))
    b1 = tl.load(b1_ptr + rl1, mask=rl1 < L1, other=0.0)
    h1 = tl.maximum(h1 + b1[None, :], 0.0)

    w2 = tl.load(w2_ptr + rl2[:, None] * L1 + rl1[None, :],
                 mask=(rl2[:, None] < L2) & (rl1[None, :] < L1), other=0.0)
    h2 = tl.dot(h1, tl.trans(w2))
    b2 = tl.load(b2_ptr + rl2, mask=rl2 < L2, other=0.0)
    h2 = tl.maximum(h2 + b2[None, :], 0.0)

    w3 = tl.load(w3_ptr + rn[:, None] * L2 + rl2[None, :],
                 mask=(rn[:, None] < N) & (rl2[None, :] < L2), other=0.0)
    h3 = tl.dot(h2, tl.trans(w3))
    b3 = tl.load(b3_ptr + rn, mask=rn < N, other=0.0)
    h3 = tl.maximum(h3 + b3[None, :], 0.0)

    tl.store(out_ptr + offs_m[:, None] * som + rn[None, :] * son,
             h3, mask=(offs_m[:, None] < M) & (rn[None, :] < N))


class BPNetNew(nn.Module):
    def __init__(self, input_dim, output_dim, level1, level2):
        super(BPNetNew, self).__init__()
        self.fc1 = nn.Linear(input_dim, level1)
        self.fc2 = nn.Linear(level1, level2)
        self.fc3 = nn.Linear(level2, output_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        orig_shape = x.shape
        K0 = orig_shape[-1]
        x2d = x.reshape(-1, K0).contiguous()
        M = x2d.shape[0]
        L1 = self.fc1.weight.shape[0]
        L2 = self.fc2.weight.shape[0]
        N = self.fc3.weight.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_M = triton.next_power_of_2(M)
        p = lambda v: max(16, triton.next_power_of_2(v))
        grid = (triton.cdiv(M, BLOCK_M),)
        _bpnet_kernel[grid](x2d, self.fc1.weight, self.fc1.bias,
                            self.fc2.weight, self.fc2.bias,
                            self.fc3.weight, self.fc3.bias, out,
                            M, K0, L1, L2, N,
                            x2d.stride(0), x2d.stride(1),
                            out.stride(0), out.stride(1),
                            BLOCK_M=BLOCK_M, PK0=p(K0), PL1=p(L1), PL2=p(L2), PN=p(N),
                            num_warps=2)
        return out.reshape(*orig_shape[:-1], N)
