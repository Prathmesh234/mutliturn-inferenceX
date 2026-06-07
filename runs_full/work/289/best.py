import torch
import triton
import triton.language as tl


@triton.jit
def _fused_split_kernel(x_ptr, first_ptr, second_ptr, n_elements,
                        CHW, LHW, S2, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    b = offs // CHW
    coff = offs % CHW
    is_first = coff < LHW
    # first output index
    first_idx = b * LHW + coff
    # second output index
    second_idx = b * S2 + (coff - LHW)
    tl.store(first_ptr + first_idx, x, mask=mask & is_first)
    tl.store(second_ptr + second_idx, x, mask=mask & (~is_first))


class SplitChannelsNew(torch.nn.Module):

    def __init__(self, split_location):
        self.split_location = split_location
        super(SplitChannelsNew, self).__init__()

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        s = x.shape
        L = self.split_location
        B = s[0]
        C = s[1]
        rest = tuple(s[2:])
        HW = 1
        for d in rest:
            HW *= d
        CHW = C * HW
        LHW = L * HW
        S2 = (C - L) * HW
        first = torch.empty((B, L) + rest, device=x.device, dtype=x.dtype)
        second = torch.empty((B, C - L) + rest, device=x.device, dtype=x.dtype)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _fused_split_kernel[grid](x, first, second, n, CHW, LHW, S2,
                                  BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return first, second

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)
