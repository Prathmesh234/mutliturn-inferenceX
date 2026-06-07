import torch
import torch.nn as nn


class Adv(nn.Module):

    def __init__(self, dim_inputs, dropout):
        super(Adv, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 32)
        self.affine2 = nn.Linear(32, 32)
        self.adv_head = nn.Linear(32, 1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.drop(self.act(self.affine1(x)))
        x = self.drop(self.act(self.affine2(x)))
        advantage = self.adv_head(x)
        return advantage


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_inputs': 4, 'dropout': 0.5}]
