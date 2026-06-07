import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1, b1, w2, b2, w3, b3, out_ptr,
                M, IN, H1, H2, OUT,
                BLOCK_M: tl.constexpr, BLOCK_IN: tl.constexpr,
                BLOCK_H1: tl.constexpr, BLOCK_H2: tl.constexpr,
                BLOCK_OUT: tl.constexpr,
                D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_in = tl.arange(0, BLOCK_IN)
    offs_h1 = tl.arange(0, BLOCK_H1)
    offs_h2 = tl.arange(0, BLOCK_H2)
    offs_out = tl.arange(0, BLOCK_OUT)

    m_mask = offs_m < M
    a = tl.load(x_ptr + offs_m[:, None] * IN + offs_in[None, :],
                mask=m_mask[:, None] & (offs_in[None, :] < IN), other=0.0)
    w1m = tl.load(w1 + offs_in[:, None] + offs_h1[None, :] * IN,
                  mask=(offs_in[:, None] < IN) & (offs_h1[None, :] < H1), other=0.0)
    acc1 = tl.dot(a, w1m)
    bias1 = tl.load(b1 + offs_h1, mask=offs_h1 < H1, other=0.0)
    acc1 = tl.maximum(acc1 + bias1[None, :], 0.0)

    w2m = tl.load(w2 + offs_h1[:, None] + offs_h2[None, :] * H1,
                  mask=(offs_h1[:, None] < H1) & (offs_h2[None, :] < H2), other=0.0)
    acc2 = tl.dot(acc1, w2m)
    bias2 = tl.load(b2 + offs_h2, mask=offs_h2 < H2, other=0.0)
    acc2 = tl.maximum(acc2 + bias2[None, :], 0.0)

    w3m = tl.load(w3 + offs_h2[:, None] + offs_out[None, :] * H2,
                  mask=(offs_h2[:, None] < H2) & (offs_out[None, :] < OUT), other=0.0)
    acc3 = tl.dot(acc2, w3m)
    bias3 = tl.load(b3 + offs_out, mask=offs_out < OUT, other=0.0)
    acc3 = acc3 + bias3[None, :]

    # softmax over dim1 of [D0, D1, D2, OUT] view of rows
    r = tl.reshape(acc3, (D0, D1, D2, BLOCK_OUT))
    r = r - tl.max(r, axis=1, keep_dims=True)
    e = tl.exp(r)
    s = tl.sum(e, axis=1, keep_dims=True)
    sm = e / s
    acc3 = tl.reshape(sm, (BLOCK_M, BLOCK_OUT))

    tl.store(out_ptr + offs_m[:, None] * OUT + offs_out[None, :], acc3,
             mask=m_mask[:, None] & (offs_out[None, :] < OUT))


class ActorNew(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNew, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        shape = x.shape
        IN = self.fc1.weight.shape[1]
        H1 = self.fc1.weight.shape[0]
        H2 = self.fc2.weight.shape[0]
        OUT = self.fc3.weight.shape[0]
        D0, D1, D2 = shape[0], shape[1], shape[2]
        x2 = x.reshape(-1, IN).contiguous()
        M = x2.shape[0]
        y2 = torch.empty((M, OUT), device=x.device, dtype=x.dtype)
        _mlp_kernel[(1,)](
            x2, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias, y2,
            M, IN, H1, H2, OUT,
            BLOCK_M=triton.next_power_of_2(M), BLOCK_IN=max(16, triton.next_power_of_2(IN)),
            BLOCK_H1=H1, BLOCK_H2=H2, BLOCK_OUT=max(16, triton.next_power_of_2(OUT)),
            D0=D0, D1=D1, D2=D2,
            num_warps=8, num_stages=3)
        return y2.reshape(*shape[:-1], OUT)
