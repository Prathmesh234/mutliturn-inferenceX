import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _channelnorm_kernel(x_ptr, out_ptr, w_ptr, b_ptr, C, S,
                        eps, HAS_AFFINE: tl.constexpr,
                        BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    base = pid_b * C * S
    offs = base + c[:, None] * S + s_offs[None, :]
    mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    Cf = C.to(tl.float32)
    mean = tl.sum(x, axis=0) / Cf
    xc = tl.where(c_mask[:, None], x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / (Cf - 1.0)
    inv = 1.0 / tl.sqrt(var + eps)
    y = xc * inv[None, :]

    if HAS_AFFINE:
        w = tl.load(w_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bb = tl.load(b_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        y = y * w[:, None] + bb[:, None]

    tl.store(out_ptr + offs, y, mask=mask)


class ChannelNormNew(nn.Module):
    def __init__(self, numFeatures, epsilon=1e-05, affine=True):
        super(ChannelNormNew, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        xc = x.contiguous()
        B = xc.shape[0]
        C = xc.shape[1]
        S = xc.numel() // (B * C)
        out = torch.empty_like(xc)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_S = min(1024, triton.next_power_of_2(S))
        grid = (B, triton.cdiv(S, BLOCK_S))
        has_affine = self.weight is not None
        w = self.weight if has_affine else xc
        bb = self.bias if has_affine else xc
        _channelnorm_kernel[grid](xc, out, w, bb, C, S,
                                  self.epsilon, has_affine, BLOCK_C, BLOCK_S,
                                  num_warps=1)
        return out
