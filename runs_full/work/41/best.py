import torch
import triton
import triton.language as tl


@triton.jit
def _normact_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < N
    offs = rows[:, None] * N + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sq = x * x
    s = tl.sum(sq, axis=1)
    out = sq / s[:, None]
    tl.store(out_ptr + offs, out, mask=mask)


class NormActivationNew(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        ndim = tensor.dim()
        dim = self.dim % ndim
        if dim == ndim - 1:
            x = tensor.contiguous()
            N = x.shape[-1]
            M = x.numel() // N
            out = torch.empty_like(x)
            BLOCK_N = triton.next_power_of_2(N)
            BLOCK_M = 64
            grid = (triton.cdiv(M, BLOCK_M),)
            _normact_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1)
            return out
        perm = [i for i in range(ndim) if i != dim] + [dim]
        x = tensor.permute(*perm).contiguous()
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _normact_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1)
        out = out.view(*x.shape)
        inv = [0] * ndim
        for i, p in enumerate(perm):
            inv[p] = i
        return out.permute(*inv).contiguous()
