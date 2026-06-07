import torch
import torch.nn as nn
from torch.nn import Parameter
import triton
import triton.language as tl


@triton.jit
def _cos_attn_kernel(nb_ptr, sf_ptr, wn_ptr, ws_ptr, out_ptr,
                     M, C: tl.constexpr, NUM_HEADS,
                     C_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    head = rows % NUM_HEADS
    cols = tl.arange(0, C_POW2)
    col_mask = cols < C
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None] * C + cols[None, :]
    nb = tl.load(nb_ptr + offs, mask=mask, other=0.0)
    sf = tl.load(sf_ptr + offs, mask=mask, other=0.0)
    w_offs = head[:, None] * C + cols[None, :]
    wn = tl.load(wn_ptr + w_offs, mask=mask, other=0.0)
    ws = tl.load(ws_ptr + w_offs, mask=mask, other=0.0)
    acc = tl.sum(nb * sf * wn * ws, axis=1)
    tl.store(out_ptr + rows, acc, mask=row_mask)


class CosAttentionNew(nn.Module):
    def __init__(self, num_heads, out_channels):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.att_self_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, self.num_heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, neighbor_vecs, self_vecs):
        C = self.out_channels
        out_shape = neighbor_vecs.shape[:-1]
        nb = neighbor_vecs.reshape(-1, C)
        sf = self_vecs.reshape(-1, C)
        M = nb.shape[0]
        out = torch.empty(M, device=nb.device, dtype=nb.dtype)
        C_POW2 = triton.next_power_of_2(C)
        BLOCK_M = triton.next_power_of_2(M)
        grid = (1,)
        _cos_attn_kernel[grid](nb, sf, self.att_neighbor_weight, self.att_self_weight,
                               out, M, C, self.num_heads,
                               C_POW2=C_POW2, BLOCK_M=BLOCK_M, num_warps=1)
        return out.view(out_shape)
