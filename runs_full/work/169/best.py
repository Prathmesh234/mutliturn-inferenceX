import torch
import numpy as np
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _linear_act(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                stride_xm, stride_xk, stride_wn, stride_wk,
                stride_om, stride_on,
                ACT: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(w))
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    if ACT == 0:  # leaky relu 0.2
        acc = tl.where(acc >= 0, acc, acc * 0.2)
    elif ACT == 1:  # sigmoid
        acc = 1.0 / (1.0 + tl.exp(-acc))
    o = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _linear(x, w, b, act):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_act[grid](x, w, b, out, M, N, K,
                      x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                      out.stride(0), out.stride(1),
                      act, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
    return out


class DiscriminatorNew(nn.Module):

    def __init__(self, img_shape, hidden_dim=1024):
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, img):
        x = img.view(img.size(0), -1).contiguous()
        x = _linear(x, self.fc1.weight, self.fc1.bias, 0)
        x = _linear(x, self.fc2.weight, self.fc2.bias, 0)
        x = _linear(x, self.fc3.weight, self.fc3.bias, 0)
        x = _linear(x, self.fc4.weight, self.fc4.bias, 1)
        return x
