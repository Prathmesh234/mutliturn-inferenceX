import torch
import torch.nn as nn
from torchaudio import transforms


class Delta(nn.Module):

    def __init__(self, order=2, **kwargs):
        super(Delta, self).__init__()
        self.order = order
        self.compute_delta = transforms.ComputeDeltas(**kwargs)

    def forward(self, x):
        feats = [x]
        for o in range(self.order):
            feat = feats[-1].transpose(0, 1).unsqueeze(0)
            delta = self.compute_delta(feat)
            feats.append(delta.squeeze(0).transpose(0, 1))
        x = torch.cat(feats, dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
