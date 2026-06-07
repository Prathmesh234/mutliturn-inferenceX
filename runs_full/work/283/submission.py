import torch
import triton
import triton.language as tl


@triton.jit
def _tv_kernel(img_ptr, out_ptr, CHW, H, W, BLOCK_SIZE: tl.constexpr):
    n = tl.program_id(axis=0)
    base = n * CHW
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < CHW
    w = idx % W
    h = (idx // W) % H
    val = tl.load(img_ptr + base + idx, mask=mask, other=0.0)
    m1 = mask & (h < H - 1)
    v1 = tl.load(img_ptr + base + idx + W, mask=m1, other=0.0)
    a1 = tl.sum(tl.where(m1, tl.abs(v1 - val), 0.0))
    m2 = mask & (w < W - 1)
    v2 = tl.load(img_ptr + base + idx + 1, mask=m2, other=0.0)
    a2 = tl.sum(tl.where(m2, tl.abs(v2 - val), 0.0))
    tl.store(out_ptr + n, a1 + a2)


class TotalVariationNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Input type is not a torch.Tensor. Got {type(img)}')
        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(
                f'Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.')
        is3d = (img.dim() == 3)
        x = img.unsqueeze(0) if is3d else img
        x = x.contiguous()
        N, C, H, W = x.shape
        CHW = C * H * W
        out = torch.empty(N, device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(CHW)
        _tv_kernel[(N,)](x, out, CHW, H, W, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        if is3d:
            return out[0]
        return out
