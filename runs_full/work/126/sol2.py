import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr, spt_ptr, out_ptr,
    Wq_ptr, Wsk_ptr, Wsv_ptr,
    Wxu_ptr, bxu_ptr,
    Wu_ptr, bu_ptr,
    Wssq_ptr, Wssk_ptr, Wssv_ptr,
    gs_ptr, gb_ptr, bs_ptr,
    H, N, SP, scale,
    BH: tl.constexpr, BN: tl.constexpr, BSP: tl.constexpr,
):
    b = tl.program_id(0)
    oh = tl.arange(0, BH)
    on = tl.arange(0, BN)
    osp = tl.arange(0, BSP)
    mh = oh < H
    mn = on < N
    msp = osp < SP
    H2 = 2 * H

    # weight blocks [H,H]: W[out,in] at out*H+in
    Wq = tl.load(Wq_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wsk = tl.load(Wsk_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wsv = tl.load(Wsv_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wssq = tl.load(Wssq_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wssk = tl.load(Wssk_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wssv = tl.load(Wssv_ptr + oh[:, None] * H + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)

    # fc_x_update [H,2H] split into two [H,H]
    Wxu_a = tl.load(Wxu_ptr + oh[:, None] * H2 + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wxu_b = tl.load(Wxu_ptr + oh[:, None] * H2 + (H + oh[None, :]), mask=mh[:, None] & mh[None, :], other=0.0)
    bxu = tl.load(bxu_ptr + oh, mask=mh, other=0.0)

    # fc_update [2H,2H] split into 4 [H,H] + biases
    Wu_ta = tl.load(Wu_ptr + oh[:, None] * H2 + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wu_tb = tl.load(Wu_ptr + oh[:, None] * H2 + (H + oh[None, :]), mask=mh[:, None] & mh[None, :], other=0.0)
    Wu_ba = tl.load(Wu_ptr + (H + oh[:, None]) * H2 + oh[None, :], mask=mh[:, None] & mh[None, :], other=0.0)
    Wu_bb = tl.load(Wu_ptr + (H + oh[:, None]) * H2 + (H + oh[None, :]), mask=mh[:, None] & mh[None, :], other=0.0)
    bu_t = tl.load(bu_ptr + oh, mask=mh, other=0.0)
    bu_b = tl.load(bu_ptr + (H + oh), mask=mh, other=0.0)

    gs = tl.load(gs_ptr + oh, mask=mh, other=0.0)
    gb = tl.load(gb_ptr + oh, mask=mh, other=0.0)
    bs = tl.load(bs_ptr + oh, mask=mh, other=0.0)

    # spt [N,H]
    spt = tl.load(spt_ptr + on[:, None] * H + oh[None, :], mask=mn[:, None] & mh[None, :], other=0.0)

    # x[b] [H(channel), SP] -> proto_x mean
    xb = x_ptr + b * H * SP
    xblk = tl.load(xb + oh[:, None] * SP + osp[None, :], mask=mh[:, None] & msp[None, :], other=0.0)
    px = tl.sum(xblk, axis=1) / SP  # [BH]

    # Attention 1
    q1 = tl.sum(px[None, :] * Wq, axis=1)  # [BH]
    kspt = tl.sum(spt[:, None, :] * Wsk[None, :, :], axis=2)  # [BN,BH]
    vspt = tl.sum(spt[:, None, :] * Wsv[None, :, :], axis=2)
    sc1 = tl.sum(q1[None, :] * kspt, axis=1) * scale  # [BN]
    sc1 = tl.where(mn, sc1, float('-inf'))
    sc1 = sc1 - tl.max(sc1, axis=0)
    p1 = tl.exp(sc1)
    p1 = p1 / tl.sum(p1, axis=0)
    agg1 = tl.sum(p1[:, None] * vspt, axis=0)  # [BH]

    pxn = tl.sum(px[None, :] * Wxu_a, axis=1) + tl.sum(agg1[None, :] * Wxu_b, axis=1) + bxu  # [BH]

    # proto_spt = spt + pxn -> [N,H]
    ps = spt + pxn[None, :]
    q2 = tl.sum(ps[:, None, :] * Wssq[None, :, :], axis=2)  # [BN,BH]
    k2 = tl.sum(ps[:, None, :] * Wssk[None, :, :], axis=2)
    v2 = tl.sum(ps[:, None, :] * Wssv[None, :, :], axis=2)
    sc2 = tl.sum(q2[:, None, :] * k2[None, :, :], axis=2) * scale  # [BN,BN]
    sc2 = tl.where(mn[None, :], sc2, float('-inf'))
    sc2 = sc2 - tl.max(sc2, axis=1)[:, None]
    p2 = tl.exp(sc2)
    p2 = p2 / tl.sum(p2, axis=1)[:, None]
    ps2 = tl.sum(p2[:, :, None] * v2[None, :, :], axis=1)  # [BN,BH]

    # Attention 3
    q3 = tl.sum(pxn[None, :] * Wq, axis=1)  # [BH]
    k3 = tl.sum(ps2[:, None, :] * Wsk[None, :, :], axis=2)  # [BN,BH]
    v3 = tl.sum(ps2[:, None, :] * Wsv[None, :, :], axis=2)
    sc3 = tl.sum(q3[None, :] * k3, axis=1) * scale  # [BN]
    sc3 = tl.where(mn, sc3, float('-inf'))
    sc3 = sc3 - tl.max(sc3, axis=0)
    p3 = tl.exp(sc3)
    p3 = p3 / tl.sum(p3, axis=0)
    agg3 = tl.sum(p3[:, None] * v3, axis=0)  # [BH]

    film_g = tl.sum(pxn[None, :] * Wu_ta, axis=1) + tl.sum(agg3[None, :] * Wu_tb, axis=1) + bu_t
    film_b = tl.sum(pxn[None, :] * Wu_ba, axis=1) + tl.sum(agg3[None, :] * Wu_bb, axis=1) + bu_b

    gamma = film_g * gs + gb  # [BH]
    beta = film_b * bs

    outblk = gamma[:, None] * xblk + beta[:, None]
    tl.store(out_ptr + b * H * SP + oh[:, None] * SP + osp[None, :],
             outblk, mask=mh[:, None] & msp[None, :])


class AttentionModuleV2New(torch.nn.Module):

    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None,
        fc_spt_value=None, fc_x_update=None, fc_update=None,
        fc_spt_spt_query=None, fc_spt_spt_key=None, fc_spt_spt_value=None,
        gamma_scale_gate=None, gamma_bias_gate=None, beta_scale_gate=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_x_query = fc_x_query if fc_x_query is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_key = fc_spt_key if fc_spt_key is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_value = fc_spt_value if fc_spt_value is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_x_update = fc_x_update if fc_x_update is not None else torch.nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.fc_update = fc_update if fc_update is not None else torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.fc_spt_spt_query = fc_spt_spt_query if fc_spt_spt_query is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_key = fc_spt_spt_key if fc_spt_spt_key is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_value = fc_spt_spt_value if fc_spt_spt_value is not None else torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma_scale_gate = gamma_scale_gate if gamma_scale_gate is not None else torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))
        self.gamma_bias_gate = gamma_bias_gate if gamma_bias_gate is not None else torch.nn.Parameter(torch.ones(size=[1, hidden_size, 1, 1, 1], requires_grad=True))
        self.beta_scale_gate = beta_scale_gate if beta_scale_gate is not None else torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

    def forward(self, x, proto_spt):
        B, C, Hs, Ws = x.shape
        H = self.hidden_size
        N = proto_spt.shape[0]
        SP = Hs * Ws
        scale = 1.0 / math.sqrt(H)
        xc = x.contiguous()
        spt = proto_spt.contiguous()
        out = torch.empty_like(xc)
        _fused_kernel[(B,)](
            xc, spt, out,
            self.fc_x_query.weight, self.fc_spt_key.weight, self.fc_spt_value.weight,
            self.fc_x_update.weight, self.fc_x_update.bias,
            self.fc_update.weight, self.fc_update.bias,
            self.fc_spt_spt_query.weight, self.fc_spt_spt_key.weight, self.fc_spt_spt_value.weight,
            self.gamma_scale_gate.reshape(-1).contiguous(),
            self.gamma_bias_gate.reshape(-1).contiguous(),
            self.beta_scale_gate.reshape(-1).contiguous(),
            H, N, SP, scale,
            BH=triton.next_power_of_2(H),
            BN=triton.next_power_of_2(N),
            BSP=triton.next_power_of_2(SP),
            num_warps=4,
        )
        return out
