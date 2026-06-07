import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(q_ptr, qw_ptr, qb_ptr, keys_ptr, attw_ptr, attb_ptr,
                  values_ptr, wout_ptr, bout_ptr, sn_ptr, out_ptr,
                  b, t_q, t_k, QS, KS, VS, OS,
                  BLOCK_B: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_TK: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_O: tl.constexpr):
    pid = tl.program_id(0)
    bi = pid // t_q
    qi = pid % t_q
    offs_b = tl.arange(0, BLOCK_B)
    bmask = offs_b < b
    offs_q = tl.arange(0, BLOCK_Q)
    qmask = offs_q < QS
    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < KS
    offs_tk = tl.arange(0, BLOCK_TK)
    tkmask = offs_tk < t_k
    offs_v = tl.arange(0, BLOCK_V)
    vmask = offs_v < VS
    offs_o = tl.arange(0, BLOCK_O)
    omask = offs_o < OS

    # att_query[b', k] = bq[k] + sum_m Wq[k,m] * query[b', qi, m]
    qb = tl.load(q_ptr + (offs_b[:, None] * t_q + qi) * QS + offs_q[None, :],
                 mask=bmask[:, None] & qmask[None, :], other=0.0)            # [B, Q]
    w = tl.load(qw_ptr + offs_k[:, None] * QS + offs_q[None, :],
                mask=kmask[:, None] & qmask[None, :], other=0.0)             # [K, Q]
    attq = tl.sum(qb[:, None, :] * w[None, :, :], axis=2)                    # [B, K]
    attq += tl.load(qb_ptr + offs_k, mask=kmask, other=0.0)[None, :]

    # keys[b', tk, k]
    keys = tl.load(keys_ptr + offs_b[:, None, None] * t_k * KS
                   + offs_tk[None, :, None] * KS + offs_k[None, None, :],
                   mask=bmask[:, None, None] & tkmask[None, :, None] & kmask[None, None, :],
                   other=0.0)                                               # [B, TK, K]
    s = attq[:, None, :] + keys                                            # [B, TK, K]
    th = 2.0 * tl.sigmoid(2.0 * s) - 1.0
    attw = tl.load(attw_ptr + offs_k, mask=kmask, other=0.0)
    attb = tl.load(attb_ptr)
    score = tl.sum(th * attw[None, None, :], axis=2) + attb                 # [B, TK]

    # softmax over batch dim
    score = tl.where(bmask[:, None] & tkmask[None, :], score, float('-inf'))
    m = tl.max(score, axis=0)
    e = tl.exp(score - m[None, :])
    denom = tl.sum(e, axis=0)
    sn_all = e / denom[None, :]                                            # [B, TK]
    sn = tl.sum(tl.where(offs_b[:, None] == bi, sn_all, 0.0), axis=0)      # [TK]
    tl.store(sn_ptr + (bi * t_q + qi) * t_k + offs_tk, sn, mask=tkmask)

    # context[v] = sum_tk sn[tk] * values[bi, tk, v]
    vals = tl.load(values_ptr + bi * t_k * VS + offs_tk[:, None] * VS + offs_v[None, :],
                   mask=tkmask[:, None] & vmask[None, :], other=0.0)        # [TK, V]
    context = tl.sum(sn[:, None] * vals, axis=0)                            # [V]

    # out = tanh(bout + Wout @ [query ; context])
    qrow = tl.load(q_ptr + (bi * t_q + qi) * QS + offs_q, mask=qmask, other=0.0)
    IN = QS + VS
    wq = tl.load(wout_ptr + offs_o[:, None] * IN + offs_q[None, :],
                 mask=omask[:, None] & qmask[None, :], other=0.0)
    wv = tl.load(wout_ptr + offs_o[:, None] * IN + (QS + offs_v[None, :]),
                 mask=omask[:, None] & vmask[None, :], other=0.0)
    acc = tl.load(bout_ptr + offs_o, mask=omask, other=0.0)
    acc += tl.sum(wq * qrow[None, :], axis=1) + tl.sum(wv * context[None, :], axis=1)
    res = 2.0 * tl.sigmoid(2.0 * acc) - 1.0
    tl.store(out_ptr + (bi * t_q + qi) * OS + offs_o, res, mask=omask)


class AttentionLayerNew(nn.Module):
    def __init__(self, query_size, key_size, value_size=None, mode='bahdanau',
                 normalize=False, dropout=0, batch_first=False, weight_norm=False,
                 output_transform=True, output_nonlinearity='tanh', output_size=None):
        super(AttentionLayerNew, self).__init__()
        assert mode == 'bahdanau' or mode == 'dot_prod'
        value_size = value_size or key_size
        self.mode = mode
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.normalize = normalize
        wn_func = lambda x: x
        if mode == 'bahdanau':
            self.linear_att = nn.Linear(key_size, 1)
            if normalize:
                self.linear_att = nn.utils.weight_norm(self.linear_att)
        if output_transform:
            output_size = output_size or query_size
            self.linear_out = wn_func(nn.Linear(query_size + key_size, output_size))
            self.output_size = output_size
        else:
            self.output_size = value_size
        self.linear_q = wn_func(nn.Linear(query_size, key_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.output_nonlinearity = output_nonlinearity
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask
        if mask is not None and not self.batch_first:
            self.mask = self.mask.t()

    def forward(self, query, keys, values=None):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if values is not None:
                values = values.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False
        values = keys if values is None else values

        query = query.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        QS = self.query_size
        KS = self.key_size
        VS = self.value_size
        OS = self.output_size

        scores_normalized = torch.empty((b, t_q, t_k), device=query.device, dtype=query.dtype)
        context = torch.empty((b, t_q, OS), device=query.device, dtype=query.dtype)
        _fused_kernel[(b * t_q,)](
            query, self.linear_q.weight, self.linear_q.bias, keys,
            self.linear_att.weight, self.linear_att.bias, values,
            self.linear_out.weight, self.linear_out.bias,
            scores_normalized, context,
            b, t_q, t_k, QS, KS, VS, OS,
            BLOCK_B=triton.next_power_of_2(b), BLOCK_Q=triton.next_power_of_2(QS),
            BLOCK_K=triton.next_power_of_2(KS), BLOCK_TK=triton.next_power_of_2(t_k),
            BLOCK_V=triton.next_power_of_2(VS), BLOCK_O=triton.next_power_of_2(OS),
            num_warps=1)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return context, scores_normalized


