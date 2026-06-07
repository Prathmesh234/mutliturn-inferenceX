import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _focal_strided(x_ptr, t_ptr, out_ptr, n,
                   d0, d1, d2, d3, d4, d5, d6, d7,
                   x0, x1, x2, x3, x4, x5, x6, x7,
                   y0, y1, y2, y3, y4, y5, y6, y7,
                   alpha, gamma, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    rem = offs
    xoff = rem - rem  # zeros, int
    toff = rem - rem

    i7 = rem % d7; rem = rem // d7; xoff += i7 * x7; toff += i7 * y7
    i6 = rem % d6; rem = rem // d6; xoff += i6 * x6; toff += i6 * y6
    i5 = rem % d5; rem = rem // d5; xoff += i5 * x5; toff += i5 * y5
    i4 = rem % d4; rem = rem // d4; xoff += i4 * x4; toff += i4 * y4
    i3 = rem % d3; rem = rem // d3; xoff += i3 * x3; toff += i3 * y3
    i2 = rem % d2; rem = rem // d2; xoff += i2 * x2; toff += i2 * y2
    i1 = rem % d1; rem = rem // d1; xoff += i1 * x1; toff += i1 * y1
    i0 = rem % d0;                  xoff += i0 * x0; toff += i0 * y0

    x = tl.load(x_ptr + xoff, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + toff, mask=mask, other=0.0).to(tl.float32)

    p = 1.0 / (1.0 + tl.exp(-x))
    one_m_p = 1.0 - p + eps
    p_eps = p + eps
    pow_one_m_p = tl.exp(gamma * tl.log(one_m_p))
    pow_p = tl.exp(gamma * tl.log(p_eps))

    term1 = -alpha * pow_one_m_p * t * tl.log(p_eps)
    term2 = (1.0 - alpha) * pow_p * (1.0 - t) * tl.log(1.0 - p + eps)
    loss = term1 - term2

    tl.store(out_ptr + offs, loss, mask=mask)


@triton.jit
def _sum_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))


class BinaryFocalLossWithLogitsNew(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogitsNew, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-08

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
                type(input)))
        if not len(input.shape) >= 2:
            raise ValueError('Invalid input shape, we expect BxCx*. Got: {}'.
                format(input.shape))
        if input.size(0) != target.size(0):
            raise ValueError(
                'Expected input batch_size ({}) to match target batch_size ({}).'
                .format(input.size(0), target.size(0)))

        inp = input.contiguous()
        tgt_base = target.contiguous()
        tgt = tgt_base.unsqueeze(dim=1)
        out_shape = torch.broadcast_shapes(inp.shape, tgt.shape)
        nd = len(out_shape)
        assert nd <= 8, "supports up to 8 dims"

        sx = list(inp.broadcast_to(out_shape).stride())
        st = list(tgt.broadcast_to(out_shape).stride())
        shp = list(out_shape)

        pad = 8 - nd
        shp = [1] * pad + shp
        sx = [0] * pad + sx
        st = [0] * pad + st

        out = torch.empty(out_shape, device=inp.device, dtype=torch.float32)
        n = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _focal_strided[grid](
            inp, tgt_base, out, n,
            shp[0], shp[1], shp[2], shp[3], shp[4], shp[5], shp[6], shp[7],
            sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], sx[7],
            st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7],
            float(self.alpha), float(self.gamma), float(self.eps),
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        loss_tmp = out.squeeze(dim=1)

        if self.reduction == 'none':
            return loss_tmp
        elif self.reduction in ('mean', 'sum'):
            flat = loss_tmp.contiguous().view(-1)
            acc = torch.zeros(1, device=flat.device, dtype=torch.float32)
            m = flat.numel()
            grid2 = (triton.cdiv(m, BLOCK_SIZE),)
            _sum_kernel[grid2](flat, acc, m, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
            s = acc[0]
            if self.reduction == 'mean':
                return s / m
            return s
        else:
            raise NotImplementedError('Invalid reduction mode: {}'.format(
                self.reduction))
