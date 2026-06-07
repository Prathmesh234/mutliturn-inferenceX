import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, mask_ptr, wa_ptr, ba_ptr, wv_ptr, bv_ptr,
                  aw_ptr, out_ptr,
                  M1: tl.constexpr, M2: tl.constexpr, M3: tl.constexpr,
                  D1: tl.constexpr, D2: tl.constexpr, H: tl.constexpr,
                  BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr):
    g = tl.program_id(0)
    a = g // M2
    c = g % M2

    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)
    dmask = offs_d < M3
    hmask = offs_h < H
    bv = tl.load(bv_ptr + 0)

    acc = tl.zeros((BLOCK_D, BLOCK_H), dtype=tl.float32)
    for b in range(M1):
        xbase = ((b * D1 + c) * D2) * H
        x = tl.load(x_ptr + xbase + offs_d[:, None] * H + offs_h[None, :],
                    mask=dmask[:, None] & hmask[None, :], other=0.0)
        s_d = tl.zeros((BLOCK_D,), dtype=tl.float32) + bv
        for j in range(H):
            wa_row = tl.load(wa_ptr + j * H + offs_h, mask=hmask, other=0.0)
            hj = tl.sum(x * wa_row[None, :], axis=1) + tl.load(ba_ptr + j)
            hj = tl.maximum(hj, 0.0)
            s_d += hj * tl.load(wv_ptr + j)
        mbase = ((a * M1 + b) * M2 + c) * M3
        m_row = tl.load(mask_ptr + mbase + offs_d, mask=dmask, other=0.0)
        full = m_row + s_d
        full = tl.where(dmask, full, float('-inf'))
        mx = tl.max(full, axis=0)
        e = tl.exp(full - mx)
        e = tl.where(dmask, e, 0.0)
        w_d = e / tl.sum(e, axis=0)
        tl.store(aw_ptr + mbase + offs_d, w_d, mask=dmask)
        acc += w_d[:, None] * x

    obase = ((a * M2 + c) * M3) * H
    tl.store(out_ptr + obase + offs_d[:, None] * H + offs_h[None, :],
             acc, mask=dmask[:, None] & hmask[None, :])


class AttentivePoolingModuleNew(nn.Module):
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModuleNew, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        batch_rep = batch_rep.contiguous()
        att_mask = att_mask.contiguous()
        D0, D1, D2, H = batch_rep.shape
        M0, M1, M2, M3 = att_mask.shape
        dev = batch_rep.device
        f32 = torch.float32
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_D = triton.next_power_of_2(M3)

        att_w_pre = torch.empty((M0, M1, M2, M3), device=dev, dtype=f32)
        utter = torch.empty((M0, M2, M3, H), device=dev, dtype=f32)
        G = M0 * M2
        _fused_kernel[(G,)](
            batch_rep.view(-1), att_mask.view(-1),
            self.W_a.weight, self.W_a.bias, self.W.weight, self.W.bias,
            att_w_pre.view(-1), utter.view(-1),
            M1=M1, M2=M2, M3=M3, D1=D1, D2=D2, H=H,
            BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, num_warps=4)

        att_w = att_w_pre.view(M0, M1, M2, M3, 1)
        return utter, att_w
