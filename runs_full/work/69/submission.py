import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _biatt_kernel(
    input_ptr, memory_ptr, win_ptr, wmem_ptr, scale_ptr, mask_ptr, out_ptr,
    I, M, D,
    HAS_MASK: tl.constexpr,
    BI: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
):
    b = tl.program_id(0)
    offs_i = tl.arange(0, BI)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    NEG = -1e30

    in_ptrs = input_ptr + b * I * D + offs_i[:, None] * D + offs_d[None, :]
    in_mask = (offs_i[:, None] < I) & (offs_d[None, :] < D)
    inp = tl.load(in_ptrs, mask=in_mask, other=0.0)

    mem_ptrs = memory_ptr + b * M * D + offs_m[:, None] * D + offs_d[None, :]
    mem_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    mem = tl.load(mem_ptrs, mask=mem_mask, other=0.0)

    w_in = tl.load(win_ptr + offs_d, mask=offs_d < D, other=0.0)
    w_mem = tl.load(wmem_ptr + offs_d, mask=offs_d < D, other=0.0)
    scale = tl.load(scale_ptr + offs_d, mask=offs_d < D, other=0.0)

    input_dot = tl.sum(inp * w_in[None, :], axis=1)
    memory_dot = tl.sum(mem * w_mem[None, :], axis=1)
    input_scaled = inp * scale[None, :]
    cross = tl.sum(input_scaled[:, None, :] * mem[None, :, :], axis=2)

    att = input_dot[:, None] + memory_dot[None, :] + cross

    if HAS_MASK:
        maskv = tl.load(mask_ptr + b * M + offs_m, mask=offs_m < M, other=0.0)
        att = att - 1e30 * (1.0 - maskv[None, :])

    valid_m = offs_m[None, :] < M
    att = tl.where(valid_m, att, NEG)

    m1 = tl.max(att, axis=1)
    e1 = tl.exp(att - m1[:, None])
    denom1 = tl.sum(e1, axis=1)
    w1 = e1 / denom1[:, None]
    output_one = tl.sum(w1[:, :, None] * mem[None, :, :], axis=1)

    valid_i = offs_i < I
    mx = tl.where(valid_i, m1, NEG)
    m2 = tl.max(mx, axis=0)
    e2 = tl.exp(mx - m2)
    denom2 = tl.sum(e2, axis=0)
    w2 = e2 / denom2
    output_two = tl.sum(w2[:, None] * inp, axis=0)

    seg0 = inp
    seg1 = output_one
    seg2 = inp * output_one
    seg3 = output_two[None, :] * output_one

    D4 = 4 * D
    base = b * I * D4 + offs_i[:, None] * D4 + offs_d[None, :]
    st_mask = (offs_i[:, None] < I) & (offs_d[None, :] < D)
    tl.store(out_ptr + base + 0 * D, seg0, mask=st_mask)
    tl.store(out_ptr + base + 1 * D, seg1, mask=st_mask)
    tl.store(out_ptr + base + 2 * D, seg2, mask=st_mask)
    tl.store(out_ptr + base + 3 * D, seg3, mask=st_mask)


class BiAttentionNew(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(
            1.0 / input_size ** 0.5))

    def forward(self, input, memory, mask=None):
        input = self.dropout(input).contiguous()
        memory = self.dropout(memory).contiguous()
        bsz, input_len, D = input.shape
        memory_len = memory.shape[1]

        out = torch.empty((bsz, input_len, 4 * D), device=input.device,
                          dtype=input.dtype)

        BI = triton.next_power_of_2(input_len)
        BM = triton.next_power_of_2(memory_len)
        BD = triton.next_power_of_2(D)

        has_mask = mask is not None
        mask_t = mask.contiguous() if has_mask else input

        _biatt_kernel[(bsz,)](
            input, memory,
            self.input_linear.weight, self.memory_linear.weight, self.dot_scale,
            mask_t, out,
            input_len, memory_len, D,
            HAS_MASK=has_mask,
            BI=BI, BM=BM, BD=BD,
            num_warps=4,
        )
        return out
