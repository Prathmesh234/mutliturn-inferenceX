import torch
import triton
import triton.language as tl


@triton.jit
def _cat_kernel(x_ptr, y_ptr, out_ptr, nx, ny, Cx_inner, Cy_inner, C_inner,
                BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mx = offs < nx
    ox = (offs // Cx_inner) * C_inner + (offs % Cx_inner)
    vx = tl.load(x_ptr + offs, mask=mx, other=0.0)
    tl.store(out_ptr + ox, vx, mask=mx)
    my = offs < ny
    oy = (offs // Cy_inner) * C_inner + Cx_inner + (offs % Cy_inner)
    vy = tl.load(y_ptr + offs, mask=my, other=0.0)
    tl.store(out_ptr + oy, vy, mask=my)


class ConcatenateChannelsNew(torch.nn.Module):
    def __init__(self, split_location):
        self.split_location = split_location
        super(ConcatenateChannelsNew, self).__init__()

    def forward(self, x, y):
        s = x.shape
        N, Cx = s[0], s[1]
        Cy = y.shape[1]
        inner = s[2] * s[3]
        C = Cx + Cy
        out = torch.empty((N, C, s[2], s[3]), dtype=x.dtype, device=x.device)
        nx = N * Cx * inner
        ny = N * Cy * inner
        BLOCK = triton.next_power_of_2(nx if nx >= ny else ny)
        _cat_kernel[(1,)](x, y, out, nx, ny, Cx * inner, Cy * inner, C * inner,
                          BLOCK_SIZE=BLOCK, num_warps=2, num_stages=1)
        return out
