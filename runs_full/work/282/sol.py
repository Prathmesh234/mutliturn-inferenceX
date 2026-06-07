import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _focal_kernel(x_ptr, t_ptr, out_ptr, n_elements,
                  alpha, gamma, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    p = 1.0 / (1.0 + tl.exp(-x))                      # sigmoid

    one_m_p = 1.0 - p + eps
    p_eps = p + eps

    # a ** gamma  ==  exp(gamma * log(a))   (a > 0 guaranteed by eps)
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
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, s)


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
        tgt = target.unsqueeze(dim=1)
        out_shape = torch.broadcast_shapes(inp.shape, tgt.shape)

        # Materialize broadcast operands (memory glue); focal math runs in Triton.
        x = inp.broadcast_to(out_shape).contiguous()
        t = tgt.broadcast_to(out_shape).contiguous()

        out = torch.empty(out_shape, device=x.device, dtype=torch.float32)
        n = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _focal_kernel[grid](x, t, out, n, float(self.alpha), float(self.gamma),
                            float(self.eps), BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

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
