import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gen_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, D, V,
                BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    mask_n = offs_n < N
    mask_d = offs_d < D
    mask_v = offs_v < V

    # x [BLOCK_N, BLOCK_D]
    x = tl.load(x_ptr + offs_n[:, None] * D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    # w [BLOCK_V, BLOCK_D]
    w = tl.load(w_ptr + offs_v[:, None] * D + offs_d[None, :],
                mask=mask_v[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    # logits [BLOCK_N, BLOCK_V] = x @ w.T + b
    logits = tl.dot(x, tl.trans(w)) + b[None, :]
    logits = tl.where(mask_v[None, :], logits, -float('inf'))
    m = tl.max(logits, axis=1)
    z = logits - m[:, None]
    s = tl.sum(tl.exp(z), axis=1)
    out = z - tl.log(s)[:, None]
    tl.store(out_ptr + offs_n[:, None] * V + offs_v[None, :], out,
             mask=mask_n[:, None] & mask_v[None, :])


class GeneratorNew(nn.Module):
    def __init__(self, d_model, vocab):
        super(GeneratorNew, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        D = self.proj.in_features
        V = self.proj.out_features
        x2 = x.contiguous().view(-1, D)
        N = x2.shape[0]
        out = torch.empty((N, V), device=x.device, dtype=x.dtype)
        BLOCK_D = max(triton.next_power_of_2(D), 16)
        BLOCK_V = max(triton.next_power_of_2(V), 16)
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N),)
        _gen_kernel[grid](x2, self.proj.weight, self.proj.bias, out, N, D, V,
                          BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=4)
        return out.view(*x.shape[:-1], V)
