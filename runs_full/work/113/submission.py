import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _amsoftmax_kernel(x_ptr, W_ptr, lbl_ptr, out_ptr,
                      B, H, S, s, m,
                      BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, BLOCK_H)
    offs_j = tl.arange(0, BLOCK_S)
    mask_b = offs_b < B
    mask_h = offs_h < H
    mask_j = offs_j < S

    # x block (B, H)
    x = tl.load(x_ptr + offs_b[:, None] * H + offs_h[None, :],
                mask=mask_b[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    xnorm = tl.sqrt(tl.sum(x * x, axis=1))           # (B,)
    x_n = x / xnorm[:, None]

    # W block (H, S)
    W = tl.load(W_ptr + offs_h[:, None] * S + offs_j[None, :],
                mask=mask_h[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    wcol_norm = tl.sqrt(tl.sum(W * W, axis=0))        # (S,)

    # wf (B, S) = x_n @ W / wcol_norm
    wf = tl.dot(x_n, W) / wcol_norm[None, :]
    wf = tl.where(mask_j[None, :], wf, -float('inf'))

    y = tl.load(lbl_ptr + offs_b, mask=mask_b, other=0)   # (B,)
    sel = offs_j[None, :] == y[:, None]
    wf_y = tl.sum(tl.where(sel, wf, 0.0), axis=1)         # (B,)

    num = s * (wf_y - m)
    exp_s_wf = tl.where(mask_j[None, :], tl.exp(s * wf), 0.0)
    sum_all = tl.sum(exp_s_wf, axis=1)
    denom = tl.exp(num) + (sum_all - tl.exp(s * wf_y))
    L = num - tl.log(denom)                               # (B,)
    L = tl.where(mask_b, L, 0.0)
    loss = -tl.sum(L) / B
    tl.store(out_ptr, loss)


class AMSoftmaxLossNew(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4, **kwargs):
        super(AMSoftmaxLossNew, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num),
                                    requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x_BxH, labels_B):
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num

        x = x_BxH.contiguous().to(torch.float32)
        W = self.W.contiguous().to(torch.float32)
        labels = labels_B.contiguous().to(torch.int64)

        B, H = x.shape
        S = self.speaker_num
        out = torch.empty((), device=x.device, dtype=torch.float32)

        BLOCK_B = max(16, triton.next_power_of_2(B))
        BLOCK_H = max(16, triton.next_power_of_2(H))
        BLOCK_S = max(16, triton.next_power_of_2(S))

        _amsoftmax_kernel[(1,)](x, W, labels, out,
                                B, H, S, self.s, self.m,
                                BLOCK_B=BLOCK_B, BLOCK_H=BLOCK_H, BLOCK_S=BLOCK_S,
                                num_warps=4)
        return out.to(x_BxH.dtype)
