import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, y1_ptr, y2_ptr,
                  w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
                  o1_ptr, o2_ptr, o3_ptr,
                  M, ND: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    layer = tl.program_id(0)
    pid = tl.program_id(1)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    ns = tl.arange(0, N)
    acc = tl.zeros((BLOCK_M, N), tl.float32)

    if layer == 0:
        K = ND
        for k in range(ND):
            xk = tl.load(x_ptr + rows * ND + k, mask=mask_m, other=0.0)
            wk = tl.load(w1_ptr + ns * K + k)
            acc += xk[:, None] * wk[None, :]
        b = tl.load(b1_ptr + ns)
        acc += b[None, :]
        tl.store(o1_ptr + rows[:, None] * N + ns[None, :], acc, mask=mask_m[:, None])
    elif layer == 1:
        K = ND + N
        for k in range(ND):
            xk = tl.load(x_ptr + rows * ND + k, mask=mask_m, other=0.0)
            wk = tl.load(w2_ptr + ns * K + k)
            acc += xk[:, None] * wk[None, :]
        for k in range(N):
            xk = tl.load(y1_ptr + rows * N + k, mask=mask_m, other=0.0)
            wk = tl.load(w2_ptr + ns * K + (ND + k))
            acc += xk[:, None] * wk[None, :]
        b = tl.load(b2_ptr + ns)
        acc += b[None, :]
        tl.store(o2_ptr + rows[:, None] * N + ns[None, :], acc, mask=mask_m[:, None])
    else:
        K = ND + N
        for k in range(ND):
            xk = tl.load(x_ptr + rows * ND + k, mask=mask_m, other=0.0)
            wk = tl.load(w3_ptr + ns * K + k)
            acc += xk[:, None] * wk[None, :]
        for k in range(N):
            xk = tl.load(y2_ptr + rows * N + k, mask=mask_m, other=0.0)
            wk = tl.load(w3_ptr + ns * K + (ND + k))
            acc += xk[:, None] * wk[None, :]
        b = tl.load(b3_ptr + ns)
        acc += b[None, :]
        tl.store(o3_ptr + rows[:, None] * N + ns[None, :], acc, mask=mask_m[:, None])


class LinearNew(nn.Module):
    def __init__(self, node_dim, hid_dim, num_class_l1, num_class_l2, num_class_l3):
        super(LinearNew, self).__init__()
        self.linear_l1 = nn.Linear(node_dim, num_class_l1)
        self.linear_l2 = nn.Linear(node_dim + num_class_l1, num_class_l2)
        self.linear_l3 = nn.Linear(node_dim + num_class_l2, num_class_l3)

    def forward(self, x, y1, y2):
        ND = x.shape[-1]
        N = self.linear_l1.weight.shape[0]
        M = x.numel() // ND
        o1 = torch.empty(list(x.shape[:-1]) + [N], device=x.device, dtype=x.dtype)
        o2 = torch.empty_like(o1)
        o3 = torch.empty_like(o1)
        BLOCK_M = 64
        grid = (3, triton.cdiv(M, BLOCK_M))
        _fused_kernel[grid](
            x, y1, y2,
            self.linear_l1.weight, self.linear_l1.bias,
            self.linear_l2.weight, self.linear_l2.bias,
            self.linear_l3.weight, self.linear_l3.bias,
            o1, o2, o3,
            M, ND=ND, N=N, BLOCK_M=BLOCK_M, num_warps=2, num_stages=1,
        )
        return o1, o2, o3
