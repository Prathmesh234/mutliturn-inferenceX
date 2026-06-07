import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _logits_kernel(mfeats_ptr, wattn_ptr, mask_ptr, out_ptr,
                   N, IDIM, D1, D2, MSTRIDE, BLOCK_C: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    cmask = offs_c < IDIM
    x = tl.load(mfeats_ptr + n * IDIM + offs_c, mask=cmask, other=0.0)
    w = tl.load(wattn_ptr + offs_c, mask=cmask, other=0.0)
    logit = tl.sum(x * w)
    rem = n % (D1 * D2)
    i = n // (D1 * D2)
    k = rem % D2
    m = tl.load(mask_ptr + i * MSTRIDE + k)
    logit = tl.where((1.0 - m) != 0.0, -1e32, logit)
    tl.store(out_ptr + n, logit)


@triton.jit
def _softmax_kernel(logits_ptr, attw_ptr, D0, D1, D2, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    i = pid // D2
    k = pid % D2
    offs_j = tl.arange(0, BLOCK_J)
    jmask = offs_j < D1
    n = (i * D1 + offs_j) * D2 + k
    l = tl.load(logits_ptr + n, mask=jmask, other=-1e32)
    mx = tl.max(l)
    e = tl.exp(l - mx)
    e = tl.where(jmask, e, 0.0)
    s = tl.sum(e)
    attw = e / s + 1e-13
    tl.store(attw_ptr + n, attw, mask=jmask)


@triton.jit
def _out_kernel(mfeats_ptr, attw_ptr, wout_ptr, res_ptr,
                N, IDIM, ODIM, BLOCK_C: tl.constexpr, BLOCK_O: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    cmask = offs_c < IDIM
    x = tl.load(mfeats_ptr + n * IDIM + offs_c, mask=cmask, other=0.0)
    a = tl.load(attw_ptr + n)
    xa = x * a
    offs_o = tl.arange(0, BLOCK_O)
    omask = offs_o < ODIM
    w = tl.load(wout_ptr + offs_o[:, None] * IDIM + offs_c[None, :],
                mask=omask[:, None] & cmask[None, :], other=0.0)
    res = tl.sum(w * xa[None, :], axis=1)
    tl.store(res_ptr + n * ODIM + offs_o, res, mask=omask)


class AttwNetHeadNew(nn.Module):

    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.mlp_attn = nn.Linear(idim, 1, bias=False)
        self.mlp_out = nn.Linear(idim, odim, bias=False)

    def forward(self, mfeats, mask):
        shape = mfeats.shape
        idim = shape[-1]
        d0 = shape[0]
        d1 = shape[1]
        d2 = 1
        for s in shape[2:-1]:
            d2 *= s
        odim = self.mlp_out.weight.shape[0]
        N = d0 * d1 * d2

        mfeats_c = mfeats.contiguous()
        mask_c = mask.contiguous()
        wattn = self.mlp_attn.weight.contiguous()
        wout = self.mlp_out.weight.contiguous()

        logits = torch.empty(N, device=mfeats.device, dtype=torch.float32)
        attw = torch.empty(N, device=mfeats.device, dtype=mfeats.dtype)
        res = torch.empty(d0 * d1 * d2 * odim, device=mfeats.device, dtype=mfeats.dtype)

        BLOCK_C = triton.next_power_of_2(idim)
        BLOCK_O = triton.next_power_of_2(odim)
        BLOCK_J = triton.next_power_of_2(d1)
        mstride = mask_c.shape[1] if mask_c.dim() > 1 else 1

        _logits_kernel[(N,)](mfeats_c, wattn, mask_c, logits,
                             N, idim, d1, d2, mstride, BLOCK_C=BLOCK_C, num_warps=4)
        _softmax_kernel[(d0 * d2,)](logits, attw, d0, d1, d2,
                                    BLOCK_J=BLOCK_J, num_warps=4)
        _out_kernel[(N,)](mfeats_c, attw, wout, res,
                          N, idim, odim, BLOCK_C=BLOCK_C, BLOCK_O=BLOCK_O, num_warps=4)

        res = res.view(*shape[:-1], odim)
        attw_out = attw.view(d0, d1, d2, 1).squeeze()
        return res, attw_out
