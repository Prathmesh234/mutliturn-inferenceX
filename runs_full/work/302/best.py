import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _single_kernel(x_ptr, y_ptr, out_ptr, R, C, G, scale,
                   G_BLOCK: tl.constexpr, C_BLOCK: tl.constexpr):
    gid = tl.arange(0, G_BLOCK)
    cid = tl.arange(0, C_BLOCK)
    gmask = gid < G
    cmask = cid < C
    n = gid // R
    r = gid % R
    idx = n[:, None] * C * R + cid[None, :] * R + r[:, None]
    mask = gmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    s = tl.sum(tl.where(mask, e, 0.0), axis=1)
    ls = x - m[:, None] - tl.log(s)[:, None]
    a = r // C
    b = r % C
    yval = tl.load(y_ptr + a, mask=gmask, other=0)
    w = tl.where(yval == b, 1.0, 0.0)
    val = -w[:, None] * ls
    acc = tl.sum(tl.where(mask, val, 0.0))
    tl.store(out_ptr, acc * scale)


def _assert_no_grad(variable):
    assert not variable.requires_grad


class CrossEntropyLossTFNew(nn.Module):

    def __init__(self):
        super(CrossEntropyLossTFNew, self).__init__()

    def forward(self, Ypred, Y, W=None):
        _assert_no_grad(Y)
        Ypred = Ypred.contiguous()
        N = Ypred.shape[0]
        C = Ypred.shape[1]
        numel = Ypred.numel()
        R = numel // (N * C)
        G = N * R

        C_BLOCK = triton.next_power_of_2(C)
        G_BLOCK = triton.next_power_of_2(G)

        if W is None and R == N * C and G_BLOCK * C_BLOCK <= 65536:
            out = torch.empty((), dtype=torch.float32, device=Ypred.device)
            Yc = Y.contiguous()
            scale = float(C) / float(numel)
            _single_kernel[(1,)](Ypred, Yc, out, R, C, G, scale,
                                 G_BLOCK=G_BLOCK, C_BLOCK=C_BLOCK, num_warps=4)
            return out

        y_onehot = torch.zeros(N, C, dtype=torch.float32, device=Ypred.device)
        y_onehot.scatter_(1, Y.data.view(-1, 1), 1)
        if W is not None:
            y_onehot = y_onehot * W
        w_full = torch.broadcast_to(y_onehot, Ypred.shape).contiguous()
        lsm = torch.log_softmax(Ypred, dim=1)
        return torch.mean(-w_full * lsm) * C
