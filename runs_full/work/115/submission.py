import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gru_kernel(x_ptr, h_ptr, out_ptr,
                W_ptr, Wb_ptr, U_ptr, Ub_ptr,
                B, INPUT: tl.constexpr, H: tl.constexpr,
                BIN: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    offs_in = tl.arange(0, BIN)
    offs_h = tl.arange(0, BH)
    mask_in = offs_in < INPUT
    mask_h = offs_h < H

    x = tl.load(x_ptr + pid * INPUT + offs_in, mask=mask_in, other=0.0)
    h = tl.load(h_ptr + pid * H + offs_h, mask=mask_h, other=0.0)

    W2 = 2 * H
    # ---- g = [x,h] @ W + Wb  (cols 0..2H) ----
    # r : cols 0..H
    Wx_r = tl.load(W_ptr + offs_in[:, None] * W2 + offs_h[None, :],
                   mask=mask_in[:, None] & mask_h[None, :], other=0.0)
    Wh_r = tl.load(W_ptr + (INPUT + offs_h[:, None]) * W2 + offs_h[None, :],
                   mask=mask_h[:, None] & mask_h[None, :], other=0.0)
    r = tl.sum(x[:, None] * Wx_r, axis=0) + tl.sum(h[:, None] * Wh_r, axis=0)
    r = r + tl.load(Wb_ptr + offs_h, mask=mask_h, other=0.0)
    r = tl.sigmoid(r)

    # u : cols H..2H
    Wx_u = tl.load(W_ptr + offs_in[:, None] * W2 + (H + offs_h[None, :]),
                   mask=mask_in[:, None] & mask_h[None, :], other=0.0)
    Wh_u = tl.load(W_ptr + (INPUT + offs_h[:, None]) * W2 + (H + offs_h[None, :]),
                   mask=mask_h[:, None] & mask_h[None, :], other=0.0)
    u = tl.sum(x[:, None] * Wx_u, axis=0) + tl.sum(h[:, None] * Wh_u, axis=0)
    u = u + tl.load(Wb_ptr + (H + offs_h), mask=mask_h, other=0.0)
    u = tl.sigmoid(u)

    rh = r * h

    # ---- c = [x, r*h] @ U + Ub ----
    Ux = tl.load(U_ptr + offs_in[:, None] * H + offs_h[None, :],
                 mask=mask_in[:, None] & mask_h[None, :], other=0.0)
    Uh = tl.load(U_ptr + (INPUT + offs_h[:, None]) * H + offs_h[None, :],
                 mask=mask_h[:, None] & mask_h[None, :], other=0.0)
    c = tl.sum(x[:, None] * Ux, axis=0) + tl.sum(rh[:, None] * Uh, axis=0)
    c = c + tl.load(Ub_ptr + offs_h, mask=mask_h, other=0.0)
    tanh_c = 2.0 * tl.sigmoid(2.0 * c) - 1.0

    hout = u * h + (1.0 - u) * tanh_c
    tl.store(out_ptr + pid * H + offs_h, hout, mask=mask_h)


class GRUCellNew(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCellNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._W = nn.Parameter(torch.FloatTensor(input_size + hidden_size,
            2 * hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(input_size + hidden_size,
            hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):
        x = x.contiguous()
        h_ = h_.contiguous()
        B = x.shape[0]
        out = torch.empty_like(h_)
        BIN = triton.next_power_of_2(self.input_size)
        BH = triton.next_power_of_2(self.hidden_size)
        grid = (B,)
        _gru_kernel[grid](x, h_, out, self._W, self._W_b, self._U, self._U_b,
                          B, self.input_size, self.hidden_size, BIN, BH,
                          num_warps=1)
        return out
