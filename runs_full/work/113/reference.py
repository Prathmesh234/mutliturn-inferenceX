import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4, **kwargs):
        """
        AM Softmax Loss
        """
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num),
            requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x_BxH, labels_B):
        """
        x shape: (B, H)
        labels shape: (B)
        """
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        W = F.normalize(self.W, dim=0)
        x_BxH = F.normalize(x_BxH, dim=1)
        wf = torch.mm(x_BxH, W)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels_B]) -
            self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0
            ) for i, y in enumerate(labels_B)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s *
            excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'speaker_num': 4}]
