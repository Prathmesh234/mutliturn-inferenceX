import torch
import triton
import triton.language as tl


@triton.jit
def _contrastive_kernel(x1_ptr, x2_ptr, label_ptr, out_ptr, N, D, margin, eps,
                        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    rid = tl.arange(0, BLOCK_N)[:, None]
    cid = tl.arange(0, BLOCK_D)[None, :]
    mask = (rid < N) & (cid < D)
    idx = rid * D + cid
    a = tl.load(x1_ptr + idx, mask=mask, other=0.0)
    b = tl.load(x2_ptr + idx, mask=mask, other=0.0)
    diff = a - b + eps
    dist2 = tl.sum(diff * diff, axis=1)
    dist = tl.sqrt(dist2)[:, None]
    lab = tl.load(label_ptr + idx, mask=mask, other=0.0)
    cl = tl.maximum(margin - dist, 0.0)
    term = (1.0 - lab) * (dist * dist) + lab * (cl * cl)
    term = tl.where(mask, term, 0.0)
    s = tl.sum(tl.sum(term, axis=1), axis=0)
    tl.store(out_ptr, s / (N * D))


class ContrastiveLossNew(torch.nn.Module):
    def __init__(self, margin=0.99):
        super(ContrastiveLossNew, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        x1 = output1.contiguous()
        x2 = output2.contiguous()
        lab = label.contiguous()
        D = x1.shape[-1]
        N = x1.numel() // D
        out = torch.empty(1, device=x1.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_D = triton.next_power_of_2(D)
        _contrastive_kernel[(1,)](x1, x2, lab, out, N, D, self.margin,
                                  1e-6, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                                  num_warps=1)
        return (out).reshape([])
