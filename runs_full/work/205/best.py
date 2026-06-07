import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mse_softmax_kernel(input_ptr, target_ptr, out_ptr,
                        n_rows, C, HW, NCHW,
                        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.arange(0, BLOCK_R)
    rmask = r < n_rows
    n = r // HW
    spatial = r % HW
    base = n * C * HW + spatial
    c = tl.arange(0, BLOCK_C)
    cmask = c < C
    offs = base[:, None] + c[None, :] * HW
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(input_ptr + offs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    denom = tl.sum(e, axis=1)
    soft = e / denom[:, None]
    t = tl.load(target_ptr + offs, mask=mask, other=0.0)
    diff = soft - t
    sq = tl.where(mask, diff * diff, 0.0)
    s = tl.sum(tl.sum(sq, axis=1), axis=0)
    tl.store(out_ptr, s * (10.0 / NCHW))


class MSELossNew(nn.Module):
    def __init__(self) -> None:
        super(MSELossNew, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', w=None) -> torch.Tensor:
        input = input.contiguous()
        target = target.contiguous()
        N, C = input.shape[0], input.shape[1]
        HW = input.numel() // (N * C)
        n_rows = N * HW
        NCHW = input.numel()
        out = torch.empty(1, device=input.device, dtype=torch.float32)
        BLOCK_R = triton.next_power_of_2(n_rows)
        BLOCK_C = triton.next_power_of_2(C)
        _mse_softmax_kernel[(1,)](input, target, out, n_rows, C, HW, NCHW,
                                  BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=1)
        return out.reshape([])
