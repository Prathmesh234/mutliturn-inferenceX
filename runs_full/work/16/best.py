import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_mlp_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    M, K, H, N,
    BLOCK_M: tl.constexpr,
    BK: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    offs_k = tl.arange(0, BK)
    offs_h = tl.arange(0, BH)
    offs_n = tl.arange(0, BN)
    k_mask = offs_k < K
    h_mask = offs_h < H
    n_mask = offs_n < N

    # load x block [BLOCK_M, BK]
    x = tl.load(x_ptr + rows[:, None] * K + offs_k[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0)
    # w1 [BH, BK]
    w1 = tl.load(w1_ptr + offs_h[:, None] * K + offs_k[None, :],
                 mask=h_mask[:, None] & k_mask[None, :], other=0.0)
    b1 = tl.load(b1_ptr + offs_h, mask=h_mask, other=0.0)

    # h = relu(x @ w1^T + b1)  -> [BLOCK_M, BH]
    h = tl.sum(x[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
    h = tl.maximum(h, 0.0)

    # w2 [BN, BH]
    w2 = tl.load(w2_ptr + offs_n[:, None] * H + offs_h[None, :],
                 mask=n_mask[:, None] & h_mask[None, :], other=0.0)
    b2 = tl.load(b2_ptr + offs_n, mask=n_mask, other=0.0)

    out = tl.sum(h[:, None, :] * w2[None, :, :], axis=2) + b2[None, :]

    tl.store(out_ptr + rows[:, None] * N + offs_n[None, :],
             out, mask=row_mask[:, None] & n_mask[None, :])


class FullyConnectedNetNew(nn.Module):
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        orig_shape = x.shape
        K = self.fc1.in_features
        H = self.fc1.out_features
        N = self.fc2.out_features
        x2d = x.reshape(-1, K).contiguous()
        M = x2d.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        def np2(v):
            return 1 << (v - 1).bit_length()

        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _fused_mlp_kernel[grid](
            x2d, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias, out,
            M, K, H, N,
            BLOCK_M=BLOCK_M, BK=np2(K), BH=np2(H), BN=np2(N),
            num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], N)
