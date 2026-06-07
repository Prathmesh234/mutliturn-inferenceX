import math
import torch
import torch.nn as nn


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        newx = x.long()
        embeddingMat = self.lut(newx) * math.sqrt(self.d_model)
        return embeddingMat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'vocab': 4}]
