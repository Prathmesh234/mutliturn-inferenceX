import torch, triton, triton.language as tl
import torch.nn as nn

@triton.jit
def _bar(arrive_ptr, sense_ptr, G, my_sense):
    my_sense = 1 - my_sense
    old = tl.atomic_add(arrive_ptr, 1)
    if old == G - 1:
        tl.atomic_xchg(arrive_ptr, 0)
        tl.atomic_xchg(sense_ptr, my_sense)
    else:
        while tl.atomic_add(sense_ptr, 0) != my_sense:
            pass
    return my_sense

@triton.jit
def _gemv_block(bid, x_ptr, w_ptr, b_ptr, out_ptr, N, K,
                RELU: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    n = bid * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K); km = k < K
        xx = tl.load(x_ptr + k, mask=km, other=0.0)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=nm[:, None] & km[None, :], other=0.0)
        acc += tl.sum(w * xx[None, :], axis=1)
    acc += tl.load(b_ptr + n, mask=nm, other=0.0)
    if RELU: acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + n, acc, mask=nm)

@triton.jit
def _mega(x_ptr, cw, cb, w1, b1, w2, b2, w3, b3, w4, b4,
          h_ptr, a_ptr, bbuf, c_ptr, out_ptr, arrive, sense,
          G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    bid = tl.program_id(0)
    ms = 0
    # conv: 1280 outputs, strided over blocks
    OC = 5; OH = 8; OW = 8; IH = 12; IW = 12
    ky = tl.arange(0, 8); kx = tl.arange(0, 8)
    pm = (ky[:, None] < 5) & (kx[None, :] < 5)
    for i in range(bid, 1280, G):
        ox = i % OW; oy = (i // OW) % OH; oc = (i // (OW * OH)) % OC; bb = i // (OW * OH * OC)
        io = bb * IH * IW + (oy + ky)[:, None] * IW + (ox + kx)[None, :]
        wo = oc * 25 + ky[:, None] * 5 + kx[None, :]
        p = tl.load(x_ptr + io, mask=pm, other=0.0)
        wv = tl.load(cw + wo, mask=pm, other=0.0)
        tl.store(h_ptr + i, tl.sum(p * wv) + tl.load(cb + oc))
    ms = _bar(arrive, sense, G, ms)
    _gemv_block(bid, h_ptr, w1, b1, a_ptr, 400, 1280, True, BLOCK_N, BLOCK_K)
    ms = _bar(arrive, sense, G, ms)
    _gemv_block(bid, a_ptr, w2, b2, bbuf, 400, 400, True, BLOCK_N, BLOCK_K)
    ms = _bar(arrive, sense, G, ms)
    _gemv_block(bid, bbuf, w3, b3, c_ptr, 400, 400, True, BLOCK_N, BLOCK_K)
    ms = _bar(arrive, sense, G, ms)
    _gemv_block(bid, c_ptr, w4, b4, out_ptr, 400, 400, False, BLOCK_N, BLOCK_K)

class NetNew(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, (5, 5)); self.flat_features = 1280
        self.fc1 = nn.Linear(1280, 400); self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400); self.fc4 = nn.Linear(400, 400)
        self.BN = 4; self.G = 320
        self._h = torch.empty(1280, device='cuda')
        self._a = torch.empty(400, device='cuda'); self._b = torch.empty(400, device='cuda')
        self._c = torch.empty(400, device='cuda'); self._out = torch.empty(400, device='cuda')
        self._arrive = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._sense = torch.zeros(1, dtype=torch.int32, device='cuda')
    def forward(self, x):
        _mega[(self.G,)](x, self.conv1.weight, self.conv1.bias,
                         self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
                         self.fc3.weight, self.fc3.bias, self.fc4.weight, self.fc4.bias,
                         self._h, self._a, self._b, self._c, self._out,
                         self._arrive, self._sense, G=self.G, BLOCK_N=4, BLOCK_K=512, num_warps=8)
        return self._out.view(1, 400)
