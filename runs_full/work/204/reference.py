import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.optim.lr_scheduler
import torch.utils.data


class MultiHeadAttention(nn.Module):

    def __init__(self, idim, odim, nhead=1, use_bias=True):
        super(MultiHeadAttention, self).__init__()
        self.idim = idim
        self.odim = odim
        self.nheads = nhead
        self.use_bias = use_bias
        self.c_lin = nn.Linear(self.idim, self.odim * 2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0)

    def forward(self, m_feats, mask):
        """
        apply muti-head attention
        Inputs:
            m_feats: multimodal features
            mask: mask for features
        Outputs:
            updated_m: updated multimodal features
        """
        mask = mask.float()
        _B, _nseg = mask.size()
        m_k = self.v_lin(self.drop(m_feats))
        m_trans = self.c_lin(self.drop(m_feats))
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)
        new_mq = m_q
        new_mk = m_k
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i]
            m2m = mk_slice @ mq_slice.transpose(1, 2) / (self.odim // self.
                nheads) ** 0.5
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1000000000.0 if
                m2m.dtype == torch.float32 else -10000.0)
            m2m_w = F.softmax(m2m, dim=2)
            w_list.append(m2m_w)
            r = m2m_w @ mv_slice if i == 0 else torch.cat((r, m2m_w @
                mv_slice), dim=2)
        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'idim': 4, 'odim': 4}]
