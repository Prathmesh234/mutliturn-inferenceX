import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, o_ptr,
                M, D0, OUT,
                BLOCK_M: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
                D0P: tl.constexpr, OUTP: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d0 = tl.arange(0, D0P)
    offs_d1 = tl.arange(0, D1)
    offs_d2 = tl.arange(0, D2)
    offs_out = tl.arange(0, OUTP)
    mask_m = offs_m < M

    # load x [BLOCK_M, D0P]
    x = tl.load(x_ptr + offs_m[:, None] * D0 + offs_d0[None, :],
                mask=mask_m[:, None] & (offs_d0[None, :] < D0), other=0.0)

    # layer1: w1 [D1, D0]
    w1 = tl.load(w1_ptr + offs_d1[:, None] * D0 + offs_d0[None, :],
                 mask=offs_d0[None, :] < D0, other=0.0)
    h1 = tl.dot(x, tl.trans(w1))
    b1 = tl.load(b1_ptr + offs_d1)
    h1 = tl.maximum(h1 + b1[None, :], 0.0)

    # layer2: w2 [D2, D1]
    w2 = tl.load(w2_ptr + offs_d2[:, None] * D1 + offs_d1[None, :])
    h2 = tl.dot(h1, tl.trans(w2))
    b2 = tl.load(b2_ptr + offs_d2)
    h2 = tl.maximum(h2 + b2[None, :], 0.0)

    # layer3: w3 [OUT, D2]
    w3 = tl.load(w3_ptr + offs_out[:, None] * D2 + offs_d2[None, :],
                 mask=offs_out[:, None] < OUT, other=0.0)
    mu = tl.dot(h2, tl.trans(w3))
    b3 = tl.load(b3_ptr + offs_out, mask=offs_out < OUT, other=0.0)
    mu = mu + b3[None, :]

    tl.store(o_ptr + offs_m[:, None] * OUT + offs_out[None, :], mu,
             mask=mask_m[:, None] & (offs_out[None, :] < OUT))


class MuNetNew(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MuNetNew, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, out_dim)

    def forward(self, x):
        orig_shape = x.shape
        D0 = orig_shape[-1]
        OUT = self.fc_mu.weight.shape[0]
        x2 = x.reshape(-1, D0).contiguous()
        M = x2.shape[0]
        out = torch.empty((M, OUT), device=x.device, dtype=torch.float32)
        BLOCK_M = max(16, triton.next_power_of_2(M))
        D0P = max(16, triton.next_power_of_2(D0))
        OUTP = max(16, triton.next_power_of_2(OUT))
        grid = (triton.cdiv(M, BLOCK_M),)
        _mlp_kernel[grid](x2, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias,
                          self.fc_mu.weight, self.fc_mu.bias, out,
                          M, D0, OUT,
                          BLOCK_M=BLOCK_M, D1=128, D2=64, D0P=D0P, OUTP=OUTP,
                          num_warps=8)
        return out.reshape(*orig_shape[:-1], OUT)
