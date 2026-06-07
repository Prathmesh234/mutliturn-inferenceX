import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _amsoftmax_kernel(x_ptr, W_ptr, lbl_ptr, out_ptr,
                      B, H, S, s, m,
                      BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    i = tl.program_id(0)
    if i >= B:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_j = tl.arange(0, BLOCK_S)
    mask_h = offs_h < H
    mask_j = offs_j < S

    # load and normalize x row i
    x = tl.load(x_ptr + i * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    xnorm = tl.sqrt(tl.sum(x * x))
    x_n = x / xnorm  # (BLOCK_H,)

    # load W block (H, S), row-major: W[h, j] at h*S + j
    w_off = offs_h[:, None] * S + offs_j[None, :]
    w_mask = mask_h[:, None] & mask_j[None, :]
    W = tl.load(W_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    wcol_norm = tl.sqrt(tl.sum(W * W, axis=0))  # (BLOCK_S,)
    dot = tl.sum(x_n[:, None] * W, axis=0)       # (BLOCK_S,)
    wf = dot / wcol_norm                          # (BLOCK_S,)
    wf = tl.where(mask_j, wf, -float('inf'))

    y = tl.load(lbl_ptr + i)
    wf_y = tl.sum(tl.where(offs_j == y, wf, 0.0))

    num = s * (wf_y - m)
    exp_s_wf = tl.where(mask_j, tl.exp(s * wf), 0.0)
    sum_all = tl.sum(exp_s_wf)
    denom = tl.exp(num) + (sum_all - tl.exp(s * wf_y))
    L_i = num - tl.log(denom)

    tl.atomic_add(out_ptr, -L_i)


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

        out = torch.zeros(1, device=x.device, dtype=torch.float32)

        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_S = triton.next_power_of_2(S)

        grid = (B,)
        _amsoftmax_kernel[grid](x, W, labels, out,
                                B, H, S, self.s, self.m,
                                BLOCK_H=BLOCK_H, BLOCK_S=BLOCK_S,
                                num_warps=4)
        return (out / B).reshape(()).to(x_BxH.dtype)
