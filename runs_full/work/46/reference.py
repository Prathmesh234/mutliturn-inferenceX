import torch
import torch.nn.functional as F
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: 256,8,36,64
        k: 256,8,36,64
        v: 256,8,36,64
        mask: 256,1,1,36
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        """
        mask(256,1,1,36)
        attn(256,8,36,36)
        这里用到了tensor的broadcast: 两个tensor必须满足，从最后一个维度开始往前算，维度要么相等，要么为1，要么不存在
        这里的mask中间两个维度为1，可以与attn做broadcast

        将mask的行索引复制到36，得到36×36的mask矩阵，batch中共256个36*36的矩阵，1/256即batch中每个样本的mask再复制到head=8个
        每个batch中样本的mask和各自的互注意力矩阵相乘
        注意力矩阵是36*36是个混淆矩阵，表示第一个元素和其余36个元素的关系，因此mask行列转置无所谓
        """
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
            dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
            dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=
            dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None,
        dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input,
            dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output,
            enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}]
