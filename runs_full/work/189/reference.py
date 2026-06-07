import torch
import torch.nn as nn
import torch.distributed
import torch.optim.lr_scheduler
import torch.utils.data


class AttwNetHead(nn.Module):

    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.mlp_attn = nn.Linear(idim, 1, bias=False)
        self.mlp_out = nn.Linear(idim, odim, bias=False)

    def masked_softmax(self, vector: 'torch.Tensor', mask: 'torch.Tensor',
        dim: 'int'=-1, memory_efficient: 'bool'=False, mask_fill_value:
        'float'=-1e+32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).bool(),
                    mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result + 1e-13

    def mask_softmax(self, feat, mask, dim=-1):
        return self.masked_softmax(feat, mask, memory_efficient=True, dim=dim)

    def get_mask_from_sequence_lengths(self, sequence_lengths:
        'torch.Tensor', max_length: 'int'):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def forward(self, mfeats, mask):
        logits = self.mlp_attn(mfeats)
        attw = self.mask_softmax(logits, mask.unsqueeze(-1).repeat(1, 1,
            logits.shape[-1]), dim=1)
        attn_feats = mfeats * attw
        res = self.mlp_out(attn_feats)
        return res, attw.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'idim': 4, 'hdim': 4, 'odim': 4}]
