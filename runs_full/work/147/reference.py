import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Attention layer according to https://arxiv.org/abs/1409.0473.

    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, query_size, key_size, value_size=None, mode=
        'bahdanau', normalize=False, dropout=0, batch_first=False,
        weight_norm=False, output_transform=True, output_nonlinearity=
        'tanh', output_size=None):
        super(AttentionLayer, self).__init__()
        assert mode == 'bahdanau' or mode == 'dot_prod'
        value_size = value_size or key_size
        self.mode = mode
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.normalize = normalize
        wn_func = wn if weight_norm else lambda x: x
        if mode == 'bahdanau':
            self.linear_att = nn.Linear(key_size, 1)
            if normalize:
                self.linear_att = nn.utils.weight_norm(self.linear_att)
        if output_transform:
            output_size = output_size or query_size
            self.linear_out = wn_func(nn.Linear(query_size + key_size,
                output_size))
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

    def calc_score(self, att_query, att_keys):
        """
        att_query is: b x t_q x n
        att_keys is b x t_k x n
        return b x t_q x t_k scores
        """
        b, t_k, n = list(att_keys.size())
        t_q = att_query.size(1)
        if self.mode == 'bahdanau':
            att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
            att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
            sum_qk = att_query + att_keys
            sum_qk = sum_qk.view(b * t_k * t_q, n)
            out = self.linear_att(F.tanh(sum_qk)).view(b, t_q, t_k)
        elif self.mode == 'dot_prod':
            out = torch.bmm(att_query, att_keys.transpose(1, 2))
            if self.normalize:
                out.div_(n ** 0.5)
        return out

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
        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        att_query = self.linear_q(query)
        scores = self.calc_score(att_query, keys)
        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            scores.masked_fill_(mask, -1000000000000.0)
        scores_normalized = F.softmax(scores)
        scores_normalized = self.dropout(scores_normalized)
        context = torch.bmm(scores_normalized, values)
        if hasattr(self, 'linear_out'):
            context = self.linear_out(torch.cat([query, context], 2))
            if self.output_nonlinearity == 'tanh':
                context = F.tanh(context)
            elif self.output_nonlinearity == 'relu':
                context = F.relu(context, inplace=True)
        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return context, scores_normalized


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_size': 4, 'key_size': 4}]
