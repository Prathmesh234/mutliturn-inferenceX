import torch
import torch.nn as nn
from torch.nn import Parameter
import triton
import triton.language as tl


@triton.jit
def _gat_sym_kernel(self_ptr, neigh_ptr, ws_ptr, wn_ptr, out_ptr,
                    N, H, C: tl.constexpr, C_BLOCK: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK + tl.arange(0, BLOCK)
    offs_c = tl.arange(0, C_BLOCK)
    row_mask = offs_n < N
    head = offs_n % H

    in_idx = offs_n[:, None] * C + offs_c[None, :]
    w_idx = head[:, None] * C + offs_c[None, :]
    mask = row_mask[:, None] & (offs_c[None, :] < C)

    s = tl.load(self_ptr + in_idx, mask=mask, other=0.0)
    nb = tl.load(neigh_ptr + in_idx, mask=mask, other=0.0)
    ws = tl.load(ws_ptr + w_idx, mask=mask, other=0.0)
    wn = tl.load(wn_ptr + w_idx, mask=mask, other=0.0)

    s_ws = tl.sum(s * ws, axis=1)
    n_wn = tl.sum(nb * wn, axis=1)
    alpha = s_ws + n_wn
    alpha = tl.where(alpha > 0, alpha, 0.2 * alpha)
    n_ws = tl.sum(nb * ws, axis=1)
    s_wn = tl.sum(s * wn, axis=1)
    out = alpha + n_ws + s_wn

    tl.store(out_ptr + offs_n, out, mask=row_mask)


class ConstAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, neighbor_vecs, self_vecs):
        return 1


class GatAttention(ConstAttention):
    def __init__(self, num_heads, out_channels):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.att_self_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        pass


class GatSymAttentionNew(GatAttention):
    def forward(self, neighbor_vecs, self_vecs):
        C = self.out_channels
        H = self.num_heads
        out_shape = self_vecs.shape[:-1]
        N = self_vecs.numel() // C
        self_c = self_vecs.contiguous()
        neigh_c = neighbor_vecs.contiguous()
        out = torch.empty(N, device=self_vecs.device, dtype=self_vecs.dtype)
        ws = self.att_self_weight.contiguous()
        wn = self.att_neighbor_weight.contiguous()
        C_BLOCK = triton.next_power_of_2(C)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _gat_sym_kernel[grid](self_c, neigh_c, ws, wn, out,
                              N, H, C, C_BLOCK, BLOCK, num_warps=4)
        return out.view(out_shape)
