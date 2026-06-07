import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _emb_kernel(x_ptr, w_ptr, out_ptr, n_idx, d_model, scale,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_idx * d_model
    mask = offs < total
    row = offs // d_model
    col = offs % d_model
    xf = tl.load(x_ptr + row, mask=mask, other=0.0)
    idx = xf.to(tl.int32)
    w = tl.load(w_ptr + idx * d_model + col, mask=mask, other=0.0)
    tl.store(out_ptr + offs, w * scale, mask=mask)


class EmbeddingsNew(nn.Module):
    def __init__(self, d_model, vocab):
        super(EmbeddingsNew, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        xf = x.contiguous().view(-1)
        n_idx = xf.numel()
        d_model = self.d_model
        out = torch.empty((n_idx, d_model), device=x.device,
                          dtype=self.lut.weight.dtype)
        scale = math.sqrt(d_model)
        total = n_idx * d_model
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        _emb_kernel[grid](xf, self.lut.weight, out, n_idx, d_model, scale,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.view(*x.shape, d_model)
