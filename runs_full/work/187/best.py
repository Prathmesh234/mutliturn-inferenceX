import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr,
                  w1a, b1a, w2a, b2a, w3a, b3a,
                  w1v, b1v, w2v, b2v, w3v, b3v,
                  out_ptr, M, S, H1, H2, A,
                  BM: tl.constexpr, BS: tl.constexpr, BH1: tl.constexpr,
                  BH2: tl.constexpr, BA: tl.constexpr):
    rm = tl.arange(0, BM)
    rs = tl.arange(0, BS)
    rh1 = tl.arange(0, BH1)
    rh2 = tl.arange(0, BH2)
    ra = tl.arange(0, BA)

    mk_x = (rm[:, None] < M) & (rs[None, :] < S)
    x = tl.load(x_ptr + rm[:, None] * S + rs[None, :], mask=mk_x, other=0.0)

    # ---- advantage branch ----
    w = tl.load(w1a + rh1[:, None] * S + rs[None, :],
                mask=(rh1[:, None] < H1) & (rs[None, :] < S), other=0.0)
    b = tl.load(b1a + rh1, mask=rh1 < H1, other=0.0)
    h = tl.maximum(tl.dot(x, tl.trans(w), out_dtype=tl.float32) + b[None, :], 0.0)

    w = tl.load(w2a + rh2[:, None] * H1 + rh1[None, :],
                mask=(rh2[:, None] < H2) & (rh1[None, :] < H1), other=0.0)
    b = tl.load(b2a + rh2, mask=rh2 < H2, other=0.0)
    h = tl.maximum(tl.dot(h, tl.trans(w), out_dtype=tl.float32) + b[None, :], 0.0)

    w = tl.load(w3a + ra[:, None] * H2 + rh2[None, :],
                mask=(ra[:, None] < A) & (rh2[None, :] < H2), other=0.0)
    b = tl.load(b3a + ra, mask=ra < A, other=0.0)
    xa = tl.dot(h, tl.trans(w), out_dtype=tl.float32) + b[None, :]

    mk_a = (rm[:, None] < M) & (ra[None, :] < A)
    s = tl.sum(tl.where(mk_a, xa, 0.0))
    mean = s / (M * A)

    # ---- value branch ----
    w = tl.load(w1v + rh1[:, None] * S + rs[None, :],
                mask=(rh1[:, None] < H1) & (rs[None, :] < S), other=0.0)
    b = tl.load(b1v + rh1, mask=rh1 < H1, other=0.0)
    hv = tl.maximum(tl.dot(x, tl.trans(w), out_dtype=tl.float32) + b[None, :], 0.0)

    w = tl.load(w2v + rh2[:, None] * H1 + rh1[None, :],
                mask=(rh2[:, None] < H2) & (rh1[None, :] < H1), other=0.0)
    b = tl.load(b2v + rh2, mask=rh2 < H2, other=0.0)
    hv = tl.maximum(tl.dot(hv, tl.trans(w), out_dtype=tl.float32) + b[None, :], 0.0)

    w = tl.load(w3v + rh2[None, :], mask=rh2[None, :] < H2, other=0.0)  # [1,H2]
    b = tl.load(b3v)
    xv = tl.sum(hv * w, axis=1) + b  # [BM]

    out = xv[:, None] + xa - mean
    tl.store(out_ptr + rm[:, None] * A + ra[None, :], out, mask=mk_a)


class Dueling_QNetworkNew(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64,
                 fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_a = nn.Linear(state_size, fc1_units)
        self.fc2_a = nn.Linear(fc1_units, fc2_units)
        self.fc3_a = nn.Linear(fc2_units, action_size)
        self.fc1_v = nn.Linear(state_size, fc1_units)
        self.fc2_v = nn.Linear(fc1_units, fc2_units)
        self.fc3_v = nn.Linear(fc2_units, 1)

    def forward(self, state):
        S = self.fc1_a.weight.shape[1]
        H1 = self.fc1_a.weight.shape[0]
        H2 = self.fc2_a.weight.shape[0]
        A = self.fc3_a.weight.shape[0]
        orig_shape = state.shape
        x = state.contiguous().view(-1, S)
        M = x.shape[0]
        out = torch.empty((M, A), device=x.device, dtype=x.dtype)

        def np2(n):
            return max(16, triton.next_power_of_2(n))

        _fused_kernel[(1,)](
            x,
            self.fc1_a.weight, self.fc1_a.bias,
            self.fc2_a.weight, self.fc2_a.bias,
            self.fc3_a.weight, self.fc3_a.bias,
            self.fc1_v.weight, self.fc1_v.bias,
            self.fc2_v.weight, self.fc2_v.bias,
            self.fc3_v.weight, self.fc3_v.bias,
            out, M, S, H1, H2, A,
            BM=np2(M), BS=np2(S), BH1=np2(H1), BH2=np2(H2), BA=np2(A),
            num_warps=8, num_stages=2)

        out_shape = orig_shape[:-1] + (A,)
        return out.view(out_shape)
