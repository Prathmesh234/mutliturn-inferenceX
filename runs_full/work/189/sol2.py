import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(mfeats_ptr, wattn_ptr, wout_ptr, mask_ptr, attw_ptr, res_ptr,
                  IDIM, ODIM, D1, D2, MSTRIDE,
                  BLOCK_J: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_O: tl.constexpr):
    pid = tl.program_id(0)
    i = pid // D2
    k = pid % D2

    offs_j = tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, BLOCK_C)
    offs_o = tl.arange(0, BLOCK_O)
    jmask = offs_j < D1
    cmask = offs_c < IDIM
    omask = offs_o < ODIM

    n_j = (i * D1 + offs_j) * D2 + k          # [BLOCK_J]
    tile = tl.load(mfeats_ptr + n_j[:, None] * IDIM + offs_c[None, :],
                   mask=jmask[:, None] & cmask[None, :], other=0.0)  # [BJ, BC]

    w = tl.load(wattn_ptr + offs_c, mask=cmask, other=0.0)
    logit = tl.sum(tile * w[None, :], axis=1)  # [BJ]

    m = tl.load(mask_ptr + i * MSTRIDE + k)
    keep = (1.0 - m) == 0.0
    logit = tl.where(keep & jmask, logit, -1e32)

    mx = tl.max(logit)
    e = tl.exp(logit - mx)
    e = tl.where(jmask, e, 0.0)
    s = tl.sum(e)
    attw = e / s + 1e-13                        # [BJ]
    tl.store(attw_ptr + n_j, attw, mask=jmask)

    wo = tl.load(wout_ptr + offs_o[:, None] * IDIM + offs_c[None, :],
                 mask=omask[:, None] & cmask[None, :], other=0.0)  # [BO, BC]
    xa = attw[:, None] * tile                   # [BJ, BC]
    res = tl.sum(xa[:, None, :] * wo[None, :, :], axis=2)  # [BJ, BO]
    tl.store(res_ptr + n_j[:, None] * ODIM + offs_o[None, :],
             res, mask=jmask[:, None] & omask[None, :])


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

        attw = torch.empty(N, device=mfeats.device, dtype=mfeats.dtype)
        res = torch.empty(N * odim, device=mfeats.device, dtype=mfeats.dtype)

        BLOCK_C = triton.next_power_of_2(idim)
        BLOCK_O = triton.next_power_of_2(odim)
        BLOCK_J = triton.next_power_of_2(d1)
        mstride = mask_c.shape[1] if mask_c.dim() > 1 else 1

        _fused_kernel[(d0 * d2,)](mfeats_c, wattn, wout, mask_c, attw, res,
                                  idim, odim, d1, d2, mstride,
                                  BLOCK_J=BLOCK_J, BLOCK_C=BLOCK_C, BLOCK_O=BLOCK_O,
                                  num_warps=1)

        res = res.view(*shape[:-1], odim)
        attw_out = attw.view(d0, d1, d2, 1).squeeze()
        return res, attw_out
