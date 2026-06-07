import torch
import torch.nn as nn
from torch.nn import Parameter
import triton
import triton.language as tl


@triton.jit
def _gat_kernel(self_ptr, nb_ptr, sw_ptr, nw_ptr, out_ptr,
                M, H, C, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    head = rows % H
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    s_ptrs = self_ptr + rows[:, None] * C + offs_c[None, :]
    n_ptrs = nb_ptr + rows[:, None] * C + offs_c[None, :]
    mask = row_mask[:, None] & c_mask[None, :]
    s = tl.load(s_ptrs, mask=mask, other=0.0)
    n = tl.load(n_ptrs, mask=mask, other=0.0)
    sw = tl.load(sw_ptr + head[:, None] * C + offs_c[None, :], mask=mask, other=0.0)
    nw = tl.load(nw_ptr + head[:, None] * C + offs_c[None, :], mask=mask, other=0.0)
    alpha = tl.sum(s * sw + n * nw, axis=1)
    alpha = tl.where(alpha >= 0, alpha, alpha * 0.2)
    tl.store(out_ptr + rows, alpha, mask=row_mask)


class GatAttentionNew(nn.Module):
    def __init__(self, num_heads, out_channels):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.att_self_weight = Parameter(torch.Tensor(1, num_heads, out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, num_heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, neighbor_vecs, self_vecs):
        C = self.out_channels
        H = self.num_heads
        out_shape = self_vecs.shape[:-1]
        s = self_vecs.contiguous().view(-1, C)
        n = neighbor_vecs.contiguous().view(-1, C)
        M = s.shape[0]
        out = torch.empty(M, device=s.device, dtype=s.dtype)
        BLOCK_M = 64
        BLOCK_C = triton.next_power_of_2(C)
        grid = (triton.cdiv(M, BLOCK_M),)
        _gat_kernel[grid](s, n, self.att_self_weight, self.att_neighbor_weight, out,
                          M, H, C, BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=1)
        return out.view(out_shape)
