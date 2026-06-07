import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_dim1_kernel(
    input_ptr, output_ptr,
    D0, D1, D2,
    BLOCK_D2: tl.constexpr,
):
    pid_d0 = tl.program_id(1)
    pid_d2 = tl.program_id(0)

    d2_off = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    mask = d2_off < D2

    acc = tl.zeros([BLOCK_D2], dtype=tl.float32)
    base = input_ptr + pid_d0 * D1 * D2
    for j in range(D1):
        vals = tl.load(base + j * D2 + d2_off, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    tl.store(output_ptr + pid_d0 * D2 + d2_off, acc, mask=mask)


@triton.jit
def sum_dim1_kernel_nomask(
    input_ptr, output_ptr,
    D0, D1, D2,
    BLOCK_D2: tl.constexpr,
):
    pid_d0 = tl.program_id(1)
    pid_d2 = tl.program_id(0)

    d2_off = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)

    acc = tl.zeros([BLOCK_D2], dtype=tl.float32)
    base = input_ptr + pid_d0 * D1 * D2
    for j in range(D1):
        vals = tl.load(base + j * D2 + d2_off)
        acc += vals.to(tl.float32)

    tl.store(output_ptr + pid_d0 * D2 + d2_off, acc)


class SumAggregatorNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, neighbor):
        shape = neighbor.shape
        D0 = int(shape[0])
        D1 = int(shape[1])
        D2 = neighbor.numel() // (D0 * D1)

        x = neighbor.contiguous()
        out_shape = (shape[0],) + shape[2:]
        out = torch.empty(out_shape, device=neighbor.device, dtype=neighbor.dtype)

        BLOCK_D2 = max(triton.next_power_of_2(D2), 32)
        grid = (triton.cdiv(D2, BLOCK_D2), D0)

        if D2 == BLOCK_D2:
            sum_dim1_kernel_nomask[grid](
                x, out, D0, D1, D2, BLOCK_D2=BLOCK_D2,
            )
        else:
            sum_dim1_kernel[grid](
                x, out, D0, D1, D2, BLOCK_D2=BLOCK_D2,
            )

        return out
