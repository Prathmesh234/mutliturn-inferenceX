import torch
import torch.nn as nn
import torch.utils.data


class PositionwiseFeedForward(nn.Module):
    """ Point-wise Feed-Forward NN, FFN, in fact 1-d convolution """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        initialization of required functions
        :param d_model: model size
        :param d_ff: intermediate size
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        run FFN
        :param x: input
        :return: output
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
