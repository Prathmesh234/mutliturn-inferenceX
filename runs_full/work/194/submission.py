import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, o_ptr,
    M, K0, N1, N2, N3,
    sx0, sx1, so0, so1,
    BM: tl.constexpr, K0P: tl.constexpr, N1P: tl.constexpr,
    N2P: tl.constexpr, N3P: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rk0 = tl.arange(0, K0P)
    rn1 = tl.arange(0, N1P)
    rn2 = tl.arange(0, N2P)
    rn3 = tl.arange(0, N3P)

    # layer 1
    x = tl.load(x_ptr + rm[:, None] * sx0 + rk0[None, :] * sx1,
                mask=(rm[:, None] < M) & (rk0[None, :] < K0), other=0.0)
    w1 = tl.load(w1_ptr + rn1[:, None] * K0 + rk0[None, :],
                 mask=(rn1[:, None] < N1) & (rk0[None, :] < K0), other=0.0)
    h1 = tl.dot(x, tl.trans(w1), allow_tf32=False)
    b1 = tl.load(b1_ptr + rn1, mask=rn1 < N1, other=0.0)
    h1 += b1[None, :]
    h1 = tl.where(h1 >= 0, h1, h1 * 0.2)

    # layer 2
    w2 = tl.load(w2_ptr + rn2[:, None] * N1 + rn1[None, :],
                 mask=(rn2[:, None] < N2) & (rn1[None, :] < N1), other=0.0)
    h2 = tl.dot(h1, tl.trans(w2), allow_tf32=False)
    b2 = tl.load(b2_ptr + rn2, mask=rn2 < N2, other=0.0)
    h2 += b2[None, :]
    h2 = tl.where(h2 >= 0, h2, h2 * 0.2)

    # layer 3
    w3 = tl.load(w3_ptr + rn3[:, None] * N2 + rn2[None, :],
                 mask=(rn3[:, None] < N3) & (rn2[None, :] < N2), other=0.0)
    h3 = tl.dot(h2, tl.trans(w3), allow_tf32=False)
    b3 = tl.load(b3_ptr + rn3, mask=rn3 < N3, other=0.0)
    h3 += b3[None, :]
    h3 = 1.0 / (1.0 + tl.exp(-h3))

    tl.store(o_ptr + rm[:, None] * so0 + rn3[None, :] * so1, h3,
             mask=(rm[:, None] < M) & (rn3[None, :] < N3))


class ClassifierNew(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        shape = input.shape
        x = input.reshape(-1, shape[-1]).contiguous().float()
        M, K0 = x.shape
        N1 = self.fc1.weight.shape[0]
        N2 = self.fc2.weight.shape[0]
        N3 = self.fc3.weight.shape[0]
        out = torch.empty((M, N3), device=x.device, dtype=torch.float32)
        BM = triton.next_power_of_2(M)
        grid = (triton.cdiv(M, BM),)
        _mlp_kernel[grid](
            x, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias, out,
            M, K0, N1, N2, N3,
            x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            BM=BM, K0P=max(16, triton.next_power_of_2(K0)),
            N1P=triton.next_power_of_2(N1), N2P=triton.next_power_of_2(N2),
            N3P=max(16, triton.next_power_of_2(N3)),
            num_warps=32, num_stages=1,
        )
        return out.reshape(*shape[:-1], N3).to(input.dtype)
