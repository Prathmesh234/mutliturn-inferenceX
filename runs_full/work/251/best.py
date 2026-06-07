import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr,
                  w1a_ptr, b1a_ptr, w1v_ptr, b1v_ptr,
                  w2a_ptr, b2a_ptr, w2v_ptr, b2v_ptr,
                  out_ptr,
                  K, H, AN,
                  stride_xm,
                  BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_AN: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    offs_a = tl.arange(0, BLOCK_AN)
    mask_k = offs_k < K
    mask_h = offs_h < H
    mask_a = offs_a < AN

    x = tl.load(x_ptr + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0).to(tl.float32)

    # fc1_a: a_hidden = relu(x @ W1a^T + b1a)  ; W1a [H, K]
    w1a = tl.load(w1a_ptr + offs_h[:, None] * K + offs_k[None, :],
                  mask=mask_h[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    a_h = tl.sum(w1a * x[None, :], axis=1) + tl.load(b1a_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    a_h = tl.maximum(a_h, 0.0)

    w1v = tl.load(w1v_ptr + offs_h[:, None] * K + offs_k[None, :],
                  mask=mask_h[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    v_h = tl.sum(w1v * x[None, :], axis=1) + tl.load(b1v_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    v_h = tl.maximum(v_h, 0.0)

    # fc2_a: a_out[AN] = a_h @ W2a^T + b2a  ; W2a [AN, H]
    w2a = tl.load(w2a_ptr + offs_a[:, None] * H + offs_h[None, :],
                  mask=mask_a[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    a_out = tl.sum(w2a * a_h[None, :], axis=1) + tl.load(b2a_ptr + offs_a, mask=mask_a, other=0.0).to(tl.float32)

    # fc2_v: scalar = v_h @ W2v^T + b2v ; W2v [1, H]
    w2v = tl.load(w2v_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    v_out = tl.sum(v_h * w2v, axis=0) + tl.load(b2v_ptr).to(tl.float32)

    mean = tl.sum(tl.where(mask_a, a_out, 0.0), axis=0) / AN
    out = a_out + v_out - mean
    tl.store(out_ptr + pid_m * AN + offs_a, out, mask=mask_a)


class duelingdqnNetNew(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM):
        super(duelingdqnNetNew, self).__init__()
        self.ACTION_NUM = ACTION_NUM
        self.fc1_a = nn.Linear(in_features=STATE_NUM, out_features=512)
        self.fc1_v = nn.Linear(in_features=STATE_NUM, out_features=512)
        self.fc2_a = nn.Linear(in_features=512, out_features=ACTION_NUM)
        self.fc2_v = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        H = 512
        AN = self.ACTION_NUM
        out = torch.empty((M, AN), device=x.device, dtype=torch.float32)
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_AN = triton.next_power_of_2(AN)
        _fused_kernel[(M,)](
            x,
            self.fc1_a.weight, self.fc1_a.bias, self.fc1_v.weight, self.fc1_v.bias,
            self.fc2_a.weight, self.fc2_a.bias, self.fc2_v.weight, self.fc2_v.bias,
            out, K, H, AN, x.stride(0),
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H, BLOCK_AN=BLOCK_AN, num_warps=16)
        return out
