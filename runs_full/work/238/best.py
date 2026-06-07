import torch
import torch.nn as nn
import triton
import triton.language as tl


class Shifted_softplus(nn.Module):
    def __init__(self):
        super(Shifted_softplus, self).__init__()
        self.act = nn.Softplus()
        self.shift = nn.Parameter(torch.tensor([0.6931]), False)

    def forward(self, X):
        return self.act(X) - self.shift


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, gamma_ptr, beta_ptr, shift_ptr, out_ptr,
                  M, IN, OUT, eps,
                  BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr):
    row = tl.program_id(0)
    in_offs = tl.arange(0, BLOCK_IN)
    out_offs = tl.arange(0, BLOCK_OUT)
    in_mask = in_offs < IN
    out_mask = out_offs < OUT

    x = tl.load(x_ptr + row * IN + in_offs, mask=in_mask, other=0.0)
    w = tl.load(w_ptr + out_offs[:, None] * IN + in_offs[None, :],
                mask=out_mask[:, None] & in_mask[None, :], other=0.0)
    acc = tl.sum(w * x[None, :], axis=1)
    b = tl.load(b_ptr + out_offs, mask=out_mask, other=0.0)
    acc = acc + b

    cnt = OUT.to(tl.float32)
    mean = tl.sum(tl.where(out_mask, acc, 0.0)) / cnt
    diff = tl.where(out_mask, acc - mean, 0.0)
    var = tl.sum(diff * diff) / cnt
    rstd = 1.0 / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + out_offs, mask=out_mask, other=0.0)
    beta = tl.load(beta_ptr + out_offs, mask=out_mask, other=0.0)
    y = diff * rstd * gamma + beta

    shift = tl.load(shift_ptr)
    sp = tl.log(1.0 + tl.exp(-tl.abs(y))) + tl.maximum(y, 0.0)
    out = sp - shift

    tl.store(out_ptr + row * OUT + out_offs, out, mask=out_mask)


class Atom_Wise_ConvolutionNew(nn.Module):
    def __init__(self, input_feature: 'int', output_feature: 'int', dropout: 'float'=0.2, UseBN: 'bool'=True):
        super(Atom_Wise_ConvolutionNew, self).__init__()
        self.conv_weights = nn.Linear(input_feature, output_feature)
        self.batch_norm = nn.LayerNorm(output_feature)
        self.UseBN = UseBN
        self.activation = Shifted_softplus()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_feats):
        IN = self.conv_weights.in_features
        OUT = self.conv_weights.out_features
        orig_shape = node_feats.shape
        x = node_feats.reshape(-1, IN).contiguous()
        M = x.shape[0]
        out = torch.empty((M, OUT), device=x.device, dtype=x.dtype)
        BLOCK_IN = triton.next_power_of_2(IN)
        BLOCK_OUT = triton.next_power_of_2(OUT)
        grid = (M,)
        _fused_kernel[grid](
            x, self.conv_weights.weight, self.conv_weights.bias,
            self.batch_norm.weight, self.batch_norm.bias, self.activation.shift,
            out, M, IN, OUT, self.batch_norm.eps,
            BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT, num_warps=1)
        return out.reshape(*orig_shape[:-1], OUT)
