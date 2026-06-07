import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _policy_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
                   logstd_ptr, mean_ptr, ls_ptr, std_ptr,
                   M, DIN, H: tl.constexpr, DOUT: tl.constexpr,
                   BDIN: tl.constexpr, BH: tl.constexpr, BDOUT: tl.constexpr,
                   SLOPE: tl.constexpr):
    row = tl.program_id(0)
    in_off = tl.arange(0, BDIN)
    in_mask = in_off < DIN
    x = tl.load(x_ptr + row * DIN + in_off, mask=in_mask, other=0.0)

    h_off = tl.arange(0, BH)
    h_mask = h_off < H

    w1 = tl.load(w1_ptr + h_off[:, None] * DIN + in_off[None, :],
                 mask=h_mask[:, None] & in_mask[None, :], other=0.0)
    h1 = tl.sum(w1 * x[None, :], axis=1) + tl.load(b1_ptr + h_off, mask=h_mask, other=0.0)
    h1 = tl.where(h1 > 0, h1, h1 * SLOPE)

    w2 = tl.load(w2_ptr + h_off[:, None] * H + h_off[None, :],
                 mask=h_mask[:, None] & h_mask[None, :], other=0.0)
    h2 = tl.sum(w2 * h1[None, :], axis=1) + tl.load(b2_ptr + h_off, mask=h_mask, other=0.0)
    h2 = tl.where(h2 > 0, h2, h2 * SLOPE)

    out_off = tl.arange(0, BDOUT)
    out_mask = out_off < DOUT
    w3 = tl.load(w3_ptr + out_off[:, None] * H + h_off[None, :],
                 mask=out_mask[:, None] & h_mask[None, :], other=0.0)
    mean = tl.sum(w3 * h2[None, :], axis=1) + tl.load(b3_ptr + out_off, mask=out_mask, other=0.0)

    ls = tl.load(logstd_ptr + out_off, mask=out_mask, other=0.0)
    std = tl.exp(ls)

    tl.store(mean_ptr + row * DOUT + out_off, mean, mask=out_mask)
    tl.store(ls_ptr + row * DOUT + out_off, ls, mask=out_mask)
    tl.store(std_ptr + row * DOUT + out_off, std, mask=out_mask)


class PolicyNew(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(PolicyNew, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, dim_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, dim_outputs))
        self.saved_actions = []
        self.rewards = []
        self.final_value = 0
        self.act = nn.LeakyReLU()

    def forward(self, x):
        din = self.affine1.in_features
        dout = self.action_mean.out_features
        H = 64
        shape = x.shape
        x2 = x.reshape(-1, din)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        mean = torch.empty((M, dout), device=x.device, dtype=x.dtype)
        ls = torch.empty((M, dout), device=x.device, dtype=x.dtype)
        std = torch.empty((M, dout), device=x.device, dtype=x.dtype)

        def npow2(n):
            return 1 << (n - 1).bit_length()

        grid = (M,)
        _policy_kernel[grid](
            x2, self.affine1.weight, self.affine1.bias,
            self.affine2.weight, self.affine2.bias,
            self.action_mean.weight, self.action_mean.bias,
            self.action_log_std,
            mean, ls, std,
            M, din, H, dout,
            npow2(din), npow2(H), npow2(dout),
            0.01, num_warps=2,
        )
        out_shape = shape[:-1] + (dout,)
        return mean.reshape(out_shape), ls.reshape(out_shape), std.reshape(out_shape)
