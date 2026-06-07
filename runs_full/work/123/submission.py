import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sap_kernel(feat_ptr, m_ptr, w_ptr, bias_ptr, U_ptr,
                B, T, C, D, E,
                BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    dp = tl.arange(0, BLOCK_D)          # over D (d')
    ep = tl.arange(0, BLOCK_E)          # over E
    md = dp < D
    me = ep < E
    mtile = md[:, None] & me[None, :]

    wv = tl.load(w_ptr + ep, mask=me, other=0.0).to(tl.float32)   # w over e
    bias = tl.load(bias_ptr).to(tl.float32)

    acc = tl.zeros((BLOCK_D, BLOCK_E), tl.float32)
    for t in range(T):
        f_off = t * (T * C * D) + c * (C * D) + dp[:, None] * D + ep[None, :]
        x = tl.load(feat_ptr + f_off, mask=mtile, other=0.0).to(tl.float32)
        e2 = tl.exp(2.0 * x)
        tile = (e2 - 1.0) / (e2 + 1.0)                            # tanh, [D,E]

        L = tl.sum(tile * wv[None, :], axis=1) + bias            # over d', [D]
        m_off = b * (T * C * D) + t * (C * D) + c * D + dp
        mv = tl.load(m_ptr + m_off, mask=md, other=0.0).to(tl.float32)
        A = tl.where(md, mv + L, float("-inf"))
        A = A - tl.max(A, axis=0)
        ex = tl.exp(A)
        S = ex / tl.sum(ex, axis=0)                              # [D]
        acc += S[:, None] * tile

    u_off = b * (C * D * E) + c * (D * E) + dp[:, None] * E + ep[None, :]
    tl.store(U_ptr + u_off, acc, mask=mtile)


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        raise NotImplementedError


class SAPNew(nn.Module):
    def __init__(self, out_dim):
        super(SAPNew, self).__init__()
        self.act_fn = nn.Tanh()
        self.sap_layer = SelfAttentionPooling(out_dim)

    def forward(self, feature, att_mask):
        feature = feature.contiguous()
        att_mask = att_mask.contiguous()
        B, T, C, D = feature.shape
        E = D
        w = self.sap_layer.W.weight.view(-1).contiguous()
        bias = self.sap_layer.W.bias.view(-1).contiguous()
        U = torch.empty((B, C, D, E), device=feature.device, dtype=feature.dtype)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_E = triton.next_power_of_2(E)
        _sap_kernel[(B * C,)](feature, att_mask, w, bias, U,
                              B, T, C, D, E,
                              BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, num_warps=1)
        return U
