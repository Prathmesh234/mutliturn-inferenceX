import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_act_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                       M, N, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, ACT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # x: [M, K] row-major; w: [N, K] row-major (PyTorch Linear weight)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BM, BK]
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BN, BK]
    acc = tl.dot(x, tl.trans(w), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    if ACT == 0:  # sigmoid
        acc = 1.0 / (1.0 + tl.exp(-acc))
    elif ACT == 1:  # relu
        acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def _linear_act(x, w, b, act):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = max(16, triton.next_power_of_2(N))
    BLOCK_K = max(16, triton.next_power_of_2(K))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_act_kernel[grid](x, w, b, out, M, N, K,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                             ACT=act, num_warps=4)
    return out


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
        h = _linear_act(x2d, self.fc1.weight, self.fc1.bias, 0)
        h = _linear_act(h, self.fc2.weight, self.fc2.bias, 0)
        h = _linear_act(h, self.fc3.weight, self.fc3.bias, 1)
        h = _linear_act(h, self.fc_output.weight, self.fc_output.bias, 0)
        out = h.reshape(*orig_shape[:-1], 1)
        return out
