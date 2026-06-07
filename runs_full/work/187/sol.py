import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
                   stride_xm, stride_xk, stride_wn, stride_wk,
                   stride_ym, stride_yn,
                   APPLY_RELU: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
    acc = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _combine_kernel(xa_ptr, xv_ptr, out_ptr, M, A,
                    BLOCK_M: tl.constexpr, BLOCK_A: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_a = tl.arange(0, BLOCK_A)
    mask = (offs_m[:, None] < M) & (offs_a[None, :] < A)
    xa = tl.load(xa_ptr + offs_m[:, None] * A + offs_a[None, :], mask=mask, other=0.0)
    s = tl.sum(tl.where(mask, xa, 0.0))
    mean = s / (M * A)
    xv = tl.load(xv_ptr + offs_m, mask=offs_m < M, other=0.0)
    out = xv[:, None] + xa - mean
    tl.store(out_ptr + offs_m[:, None] * A + offs_a[None, :], out, mask=mask)


def _linear(x, weight, bias, relu):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = max(16, triton.next_power_of_2(N))
    BLOCK_K = max(16, triton.next_power_of_2(K))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid](x, weight, bias, y, M, N, K,
                         x.stride(0), x.stride(1),
                         weight.stride(0), weight.stride(1),
                         y.stride(0), y.stride(1),
                         APPLY_RELU=relu,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=4, num_stages=2)
    return y


class Dueling_QNetworkNew(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64,
                 fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_a = nn.Linear(state_size, fc1_units)
        self.fc2_a = nn.Linear(fc1_units, fc2_units)
        self.fc3_a = nn.Linear(fc2_units, action_size)
        self.fc1_v = nn.Linear(state_size, fc1_units)
        self.fc2_v = nn.Linear(fc1_units, fc2_units)
        self.fc3_v = nn.Linear(fc2_units, 1)

    def forward(self, state):
        state_size = self.fc1_a.weight.shape[1]
        action_size = self.fc3_a.weight.shape[0]
        orig_shape = state.shape
        x = state.contiguous().view(-1, state_size)

        x_a = _linear(x, self.fc1_a.weight, self.fc1_a.bias, True)
        x_a = _linear(x_a, self.fc2_a.weight, self.fc2_a.bias, True)
        x_a = _linear(x_a, self.fc3_a.weight, self.fc3_a.bias, False)

        x_v = _linear(x, self.fc1_v.weight, self.fc1_v.bias, True)
        x_v = _linear(x_v, self.fc2_v.weight, self.fc2_v.bias, True)
        x_v = _linear(x_v, self.fc3_v.weight, self.fc3_v.bias, False)

        M = x_a.shape[0]
        A = action_size
        out = torch.empty((M, A), device=x.device, dtype=x.dtype)
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_A = triton.next_power_of_2(A)
        _combine_kernel[(1,)](x_a, x_v.contiguous(), out, M, A,
                              BLOCK_M=BLOCK_M, BLOCK_A=BLOCK_A, num_warps=4)

        out_shape = orig_shape[:-1] + (action_size,)
        return out.view(out_shape)
