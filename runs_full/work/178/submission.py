import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layer(h, w_ptr, b_ptr, K: tl.constexpr, N: tl.constexpr,
           ACT: tl.constexpr, P: tl.constexpr):
    offs = tl.arange(0, P)
    wmask = (offs[:, None] < K) & (offs[None, :] < N)
    wT = tl.load(w_ptr + offs[None, :] * K + offs[:, None], mask=wmask, other=0.0)
    b = tl.load(b_ptr + offs, mask=offs < N, other=0.0)
    acc = tl.dot(h, wT, input_precision="ieee") + b[None, :]
    if ACT == 1:
        acc = tl.where(acc >= 0, acc, acc * 0.01)
    elif ACT == 2:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    return acc


@triton.jit
def _net_kernel(x_ptr,
                w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, wo, bo,
                out_ptr, M,
                DIN: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                S3: tl.constexpr, S4: tl.constexpr, S5: tl.constexpr,
                P: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, P)
    m_mask = offs_m < M
    xmask = m_mask[:, None] & (offs_p[None, :] < DIN)
    h = tl.load(x_ptr + offs_m[:, None] * DIN + offs_p[None, :], mask=xmask, other=0.0)
    h = _layer(h, w1, b1, DIN, S1, 1, P)
    h = _layer(h, w2, b2, S1, S2, 1, P)
    h = _layer(h, w3, b3, S2, S3, 1, P)
    h = _layer(h, w4, b4, S3, S4, 1, P)
    h = _layer(h, w5, b5, S4, S5, 1, P)
    h = _layer(h, wo, bo, S5, 1, 2, P)
    o = tl.sum(tl.where(offs_p[None, :] == 0, h, 0.0), axis=1)
    tl.store(out_ptr + offs_m, o, mask=m_mask)


class Deep_Neural_NetworkNew(nn.Module):
    def __init__(self, D_in, fc1_size=40, fc2_size=20, fc3_size=40,
                 fc4_size=20, fc5_size=40):
        super(Deep_Neural_NetworkNew, self).__init__()
        self.fc1 = nn.Linear(D_in, fc1_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        nn.init.kaiming_normal_(self.fc5.weight)
        self.relu5 = nn.LeakyReLU()
        self.fc_output = nn.Linear(fc5_size, 1)
        self.fc_output_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        orig_shape = x.shape
        D_in = orig_shape[-1]
        xf = x.reshape(-1, D_in).contiguous()
        M = xf.shape[0]
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        P = 64
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _net_kernel[grid](
            xf,
            self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias, self.fc4.weight, self.fc4.bias,
            self.fc5.weight, self.fc5.bias, self.fc_output.weight, self.fc_output.bias,
            out, M,
            D_in, self.fc1.out_features, self.fc2.out_features,
            self.fc3.out_features, self.fc4.out_features, self.fc5.out_features,
            P=P, BLOCK_M=BLOCK_M, num_warps=2)
        return out.reshape(*orig_shape[:-1], 1)
