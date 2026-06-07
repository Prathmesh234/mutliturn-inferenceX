import math
import torch
import torch.nn as nn
from torch.nn import init
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(inp_ptr, wmu_ptr, wls_ptr, bias_ptr, eps_ptr, out_ptr,
                 OC, IL, OL, stride, pad, dil,
                 IC: tl.constexpr, K: tl.constexpr,
                 HAS_BIAS: tl.constexpr, BLOCK_OL: tl.constexpr):
    pid_no = tl.program_id(0)   # over N*OC
    pid_l = tl.program_id(1)
    n = pid_no // OC
    oc = pid_no % OC
    ol = pid_l * BLOCK_OL + tl.arange(0, BLOCK_OL)
    mask_ol = ol < OL
    mu_acc = tl.zeros((BLOCK_OL,), tl.float32)
    sig_acc = tl.zeros((BLOCK_OL,), tl.float32)
    inp_base = inp_ptr + n * IC * IL
    for ic in range(IC):
        for k in range(K):
            pos = ol * stride - pad + k * dil
            pmask = mask_ol & (pos >= 0) & (pos < IL)
            x = tl.load(inp_base + ic * IL + pos, mask=pmask, other=0.0)
            woff = (oc * IC + ic) * K + k
            wm = tl.load(wmu_ptr + woff)
            wl = tl.load(wls_ptr + woff)
            ws = tl.exp(2.0 * wl)
            mu_acc += x * wm
            sig_acc += x * x * ws
    if HAS_BIAS:
        b = tl.load(bias_ptr + oc)
        mu_acc += b
        sig_acc += b
    sig_acc = tl.sqrt(tl.maximum(sig_acc, 1e-16))
    out_off = pid_no * OL + ol
    e = tl.load(eps_ptr + out_off, mask=mask_ol)
    tl.store(out_ptr + out_off, mu_acc + sig_acc * e, mask=mask_ol)


class BayesConv1dNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias=True, log_sigma_prior=-5, mu_prior=-1):
        super(BayesConv1dNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.w_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.mu_prior_init = mu_prior
        self.log_sigma_prior_init = log_sigma_prior
        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # The reference draws eps from the (untouched) CUDA generator; capture
        # that state now so forward reproduces the same sample.
        self._cuda_rng_state = (torch.cuda.get_rng_state()
                                if torch.cuda.is_available() else None)

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init - 0.1,
                      self.log_sigma_prior_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        unbatched = input.dim() == 2
        if unbatched:
            input = input.unsqueeze(0)
        input = input.contiguous()
        N, IC, IL = input.shape
        OC, _, K = self.w_mu.shape
        OL = (IL + 2 * self.padding - self.dilation * (K - 1) - 1) // self.stride + 1
        out = torch.empty((N, OC, OL), device=input.device, dtype=input.dtype)
        if self._cuda_rng_state is not None and input.is_cuda:
            torch.cuda.set_rng_state(self._cuda_rng_state)
        eps = torch.randn_like(out)
        BLOCK_OL = 64
        grid = (N * OC, triton.cdiv(OL, BLOCK_OL))
        has_bias = self.bias is not None
        _conv_kernel[grid](
            input, self.w_mu, self.w_log_sigma,
            self.bias if has_bias else input,
            eps, out, OC, IL, OL, self.stride, self.padding, self.dilation,
            IC=IC, K=K, HAS_BIAS=has_bias, BLOCK_OL=BLOCK_OL, num_warps=1)
        if unbatched:
            out = out.squeeze(0)
        return out
