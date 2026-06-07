import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                M, K, H, N,
                stride_xm, stride_xk,
                stride_w1h, stride_w1k,
                stride_w2n, stride_w2h,
                stride_om, stride_on,
                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)

    # load x [BLOCK_M, BLOCK_K]
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    # load w1 [BLOCK_H, BLOCK_K]
    w1 = tl.load(w1_ptr + offs_h[:, None] * stride_w1h + offs_k[None, :] * stride_w1k,
                 mask=(offs_h[:, None] < H) & (offs_k[None, :] < K), other=0.0)
    # h = relu(x @ w1.T + b1)  -> [BLOCK_M, BLOCK_H]
    h = tl.dot(x, tl.trans(w1), allow_tf32=False)
    b1 = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)
    h += b1[None, :]
    h = tl.maximum(h, 0.0)

    # load w2 [BLOCK_N, BLOCK_H]
    w2 = tl.load(w2_ptr + offs_n[:, None] * stride_w2n + offs_h[None, :] * stride_w2h,
                 mask=(offs_n[:, None] < N) & (offs_h[None, :] < H), other=0.0)
    o = tl.dot(h.to(w2.dtype), tl.trans(w2), allow_tf32=False)
    b2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)
    o += b2[None, :]

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             o, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class NetNew(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM):
        super(NetNew, self).__init__()
        self.fc1 = nn.Linear(in_features=STATE_NUM, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=ACTION_NUM)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        M, K = x2.shape
        H = self.fc1.weight.shape[0]
        N = self.fc2.weight.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x2.dtype)
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_K = max(16, triton.next_power_of_2(K))
        BLOCK_H = max(16, triton.next_power_of_2(H))
        BLOCK_N = max(16, triton.next_power_of_2(N))
        grid = (triton.cdiv(M, BLOCK_M),)
        _mlp_kernel[grid](x2, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias, out,
                          M, K, H, N,
                          x2.stride(0), x2.stride(1),
                          self.fc1.weight.stride(0), self.fc1.weight.stride(1),
                          self.fc2.weight.stride(0), self.fc2.weight.stride(1),
                          out.stride(0), out.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                          BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
                          num_warps=4, num_stages=2)
        return out.reshape(*orig_shape[:-1], N)
