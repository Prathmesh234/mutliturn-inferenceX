import torch
from torch import nn


def tanh_clip(x, clip_val=10.0):
    """soft clip values to the range [-clip_val, +clip_val]"""
    if clip_val is not None:
        x_clip = clip_val * torch.tanh(1.0 / clip_val * x)
    else:
        x_clip = x
    return x_clip


class AmdimNCELoss(nn.Module):
    """Compute the NCE scores for predicting r_src->r_trg."""

    def __init__(self, tclip):
        super().__init__()
        self.tclip = tclip

    def forward(self, anchor_representations, positive_representations,
        mask_mat):
        """
        Args:
            anchor_representations: (batch_size, emb_dim)
            positive_representations: (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
            mask_mat: (n_batch_gpu, n_batch)

        Output:
            raw_scores: (n_batch_gpu, n_locs)
            nce_scores: (n_batch_gpu, n_locs)
            lgt_reg : scalar
        """
        r_src = anchor_representations
        r_trg = positive_representations
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors
            ).float()
        mask_neg = 1.0 - mask_pos
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors
            )
        raw_scores = raw_scores / emb_dim ** 0.5
        lgt_reg = 0.05 * (raw_scores ** 2.0).mean()
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)
        """
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        """
        pos_scores = (mask_pos * raw_scores).sum(dim=1)
        neg_scores = mask_neg * raw_scores - self.tclip * mask_pos
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim
            =1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) +
            neg_sumexp)
        pos_shiftexp = pos_scores - neg_maxes
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()
        return nce_scores, lgt_reg


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'tclip': 4}]
