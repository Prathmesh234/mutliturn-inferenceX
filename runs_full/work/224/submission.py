import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mul_kernel(layer_ptr, mask_ptr, out_ptr, n_elements, S3, S2,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements
    x = tl.load(layer_ptr + offs, mask=valid)
    midx = (offs // S3) % S2
    m = tl.load(mask_ptr + midx, mask=valid)
    tl.store(out_ptr + offs, x * m, mask=valid)


class Custom_dropoutNew(nn.Module):
    def __init__(self, dp_rate: 'float', n_permutation: 'int'):
        super(Custom_dropoutNew, self).__init__()
        self.dropout = nn.Dropout(p=dp_rate)
        self.ones = nn.Parameter(torch.ones(n_permutation), requires_grad=False)

    def forward(self, layer):
        m = self.dropout(self.ones)
        S0 = layer.shape[0]
        S3 = layer.shape[3] if layer.dim() >= 4 else 1
        out = torch.empty_like(layer)
        n = layer.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        grid = (1,)
        _mul_kernel[grid](layer, m, out, n, S3, S0,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
