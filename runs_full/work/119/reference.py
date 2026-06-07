import torch
import torch.nn as nn


class SoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, **kwargs):
        """
        Softmax Loss
        """
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(hidden_dim, speaker_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_BxH, labels_B):
        """
        x shape: (B, H)
        labels shape: (B)
        """
        logits_BxSpn = self.fc(x_BxH)
        loss = self.loss(logits_BxSpn, labels_B)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'speaker_num': 4}]
