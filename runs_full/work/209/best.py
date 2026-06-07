import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, w_ptr, b_ptr, out_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_wn, stride_wk,
                   HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    # out[m, n] = act( sum_k a[m,k] * w[n,k] + bias[n] )
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # w as [BLOCK_K, BLOCK_N]: w_blk[k, n] = w[n, k]
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, w, input_precision="ieee")
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    if ACT:
        acc = tl.where(acc > 0, acc, 0.01 * acc)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _autoenc_kernel(adj_ptr, emb_ptr, lmat_ptr, out_l1, out_l2, out_reg,
                    w0, b0, w1, b1, w2, b2, w3, b3,
                    M, NN, H1, H2, beta, nu1, nu2,
                    BR: tl.constexpr, BD: tl.constexpr):
    offs_r = tl.arange(0, BR)
    offs_d = tl.arange(0, BD)
    rm = offs_r < M
    dc = offs_d[:, None]
    dr = offs_d[None, :]

    adj = tl.load(adj_ptr + offs_r[:, None] * NN + dr,
                  mask=rm[:, None] & (dr < NN), other=0.0)

    w = tl.load(w0 + dr * NN + dc, mask=(dr < H1) & (dc < NN), other=0.0)
    t0 = tl.dot(adj, w, input_precision="ieee")
    t0 += tl.load(b0 + offs_d, mask=offs_d < H1, other=0.0)[None, :]
    t0 = tl.where(t0 > 0, t0, 0.01 * t0)

    w = tl.load(w1 + dr * H1 + dc, mask=(dr < H2) & (dc < H1), other=0.0)
    emb = tl.dot(t0, w, input_precision="ieee")
    emb += tl.load(b1 + offs_d, mask=offs_d < H2, other=0.0)[None, :]
    emb = tl.where(emb > 0, emb, 0.01 * emb)

    w = tl.load(w2 + dr * H2 + dc, mask=(dr < H1) & (dc < H2), other=0.0)
    d0 = tl.dot(emb, w, input_precision="ieee")
    d0 += tl.load(b2 + offs_d, mask=offs_d < H1, other=0.0)[None, :]
    d0 = tl.where(d0 > 0, d0, 0.01 * d0)

    w = tl.load(w3 + dr * H1 + dc, mask=(dr < NN) & (dc < H1), other=0.0)
    d1 = tl.dot(d0, w, input_precision="ieee")
    d1 += tl.load(b3 + offs_d, mask=offs_d < NN, other=0.0)[None, :]
    d1 = tl.where(d1 > 0, d1, 0.01 * d1)

    tl.store(emb_ptr + offs_r[:, None] * H2 + dr, emb,
             mask=rm[:, None] & (dr < H2))

    # L_1st = sum( l_mat * (emb @ emb^T) )   [the 2*alpha factor is applied in python]
    gram = tl.dot(emb, tl.trans(emb), input_precision="ieee")
    lmat = tl.load(lmat_ptr + offs_r[:, None] * M + dr,
                   mask=rm[:, None] & (dr < M), other=0.0)
    tl.atomic_add(out_l1, tl.sum(lmat * gram))

    # L_2nd = sum( ((adj - d1) * adj * beta)^2 )
    diff = (adj - d1) * adj * beta
    tl.atomic_add(out_l2, tl.sum(diff * diff))

    # L_reg = sum_params nu1*|p| + nu2*p^2  over all weights and biases
    ii = offs_d[:, None]
    jj = offs_d[None, :]
    p = tl.load(w0 + ii * NN + jj, mask=(ii < H1) & (jj < NN), other=0.0)
    reg = tl.sum(nu1 * tl.abs(p) + nu2 * p * p)
    p = tl.load(w1 + ii * H1 + jj, mask=(ii < H2) & (jj < H1), other=0.0)
    reg += tl.sum(nu1 * tl.abs(p) + nu2 * p * p)
    p = tl.load(w2 + ii * H2 + jj, mask=(ii < H1) & (jj < H2), other=0.0)
    reg += tl.sum(nu1 * tl.abs(p) + nu2 * p * p)
    p = tl.load(w3 + ii * H1 + jj, mask=(ii < NN) & (jj < H1), other=0.0)
    reg += tl.sum(nu1 * tl.abs(p) + nu2 * p * p)
    bv = tl.load(b0 + offs_d, mask=offs_d < H1, other=0.0)
    reg += tl.sum(nu1 * tl.abs(bv) + nu2 * bv * bv)
    bv = tl.load(b1 + offs_d, mask=offs_d < H2, other=0.0)
    reg += tl.sum(nu1 * tl.abs(bv) + nu2 * bv * bv)
    bv = tl.load(b2 + offs_d, mask=offs_d < H1, other=0.0)
    reg += tl.sum(nu1 * tl.abs(bv) + nu2 * bv * bv)
    bv = tl.load(b3 + offs_d, mask=offs_d < NN, other=0.0)
    reg += tl.sum(nu1 * tl.abs(bv) + nu2 * bv * bv)
    tl.atomic_add(out_reg, reg)


@triton.jit
def _mul_sum_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(a * b))


@triton.jit
def _l2nd_kernel(adj_ptr, t_ptr, out_ptr, n, beta, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    adj = tl.load(adj_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0)
    d = (adj - t) * adj * beta
    tl.atomic_add(out_ptr, tl.sum(d * d))


@triton.jit
def _reg_kernel(p_ptr, out_ptr, n, nu1, nu2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(nu1 * tl.abs(p) + nu2 * p * p))


def _linear(x, weight, bias, act):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](x, weight, bias if bias is not None else x, out,
                         M, N, K,
                         x.stride(0), x.stride(1),
                         weight.stride(0), weight.stride(1),
                         HAS_BIAS=bias is not None, ACT=act,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=4)
    return out


class SDNE_layerNew(nn.Module):

    def __init__(self, num_node, hidden_size1, hidden_size2, droput, alpha,
                 beta, nu1, nu2):
        super(SDNE_layerNew, self).__init__()
        self.num_node = num_node
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.droput = droput
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.encode0 = nn.Linear(self.num_node, self.hidden_size1)
        self.encode1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.decode0 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.decode1 = nn.Linear(self.hidden_size1, self.num_node)

    def forward(self, adj_mat, l_mat):
        adj = adj_mat.contiguous()
        l_mat = l_mat.contiguous()

        Mrows = adj.shape[0]
        maxd = max(self.num_node, self.hidden_size1, self.hidden_size2)
        acc = torch.zeros(3, device=adj.device, dtype=torch.float32)
        l1_acc, l2_acc, reg_acc = acc[0:1], acc[1:2], acc[2:3]
        fused = Mrows <= 16 and maxd <= 16
        if fused:
            emb = torch.empty((Mrows, self.hidden_size2), device=adj.device, dtype=torch.float32)
            _autoenc_kernel[(1,)](adj, emb, l_mat, l1_acc, l2_acc, reg_acc,
                                  self.encode0.weight, self.encode0.bias,
                                  self.encode1.weight, self.encode1.bias,
                                  self.decode0.weight, self.decode0.bias,
                                  self.decode1.weight, self.decode1.bias,
                                  Mrows, self.num_node, self.hidden_size1, self.hidden_size2,
                                  self.beta, self.nu1, self.nu2, BR=16, BD=16, num_warps=1)
            self.embedding = emb
        else:
            t0 = _linear(adj, self.encode0.weight, self.encode0.bias, True)
            emb = _linear(t0, self.encode1.weight, self.encode1.bias, True)
            self.embedding = emb
            d0 = _linear(emb, self.decode0.weight, self.decode0.bias, True)
            d1 = _linear(d0, self.decode1.weight, self.decode1.bias, True)
            M = emb.shape[0]
            gram = torch.empty((M, M), device=emb.device, dtype=torch.float32)
            grid = (triton.cdiv(M, 32), triton.cdiv(M, 32))
            _matmul_kernel[grid](emb, emb, emb, gram, M, M, emb.shape[1],
                                 emb.stride(0), emb.stride(1),
                                 emb.stride(0), emb.stride(1),
                                 HAS_BIAS=False, ACT=False,
                                 BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4)
            _mul_sum_kernel[(triton.cdiv(gram.numel(), 1024),)](l_mat, gram, l1_acc,
                                                                gram.numel(), BLOCK=1024)
            _l2nd_kernel[(triton.cdiv(adj.numel(), 1024),)](adj, d1, l2_acc,
                                                            adj.numel(), self.beta, BLOCK=1024)

        if not fused:
            flat = torch.cat([p.reshape(-1) for p in self.parameters()])
            npar = flat.numel()
            _reg_kernel[(triton.cdiv(npar, 1024),)](flat, reg_acc, npar,
                                                    self.nu1, self.nu2, BLOCK=1024)

        L_1st = 2.0 * l1_acc[0]
        L_2nd = l2_acc[0]
        L_reg = reg_acc[0]

        return self.alpha * L_1st, L_2nd, self.alpha * L_1st + L_2nd, L_reg
