import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self) ->None:
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', w=None
        ) ->torch.Tensor:
        input_soft = F.softmax(input, dim=1)
        return self.mse_loss(input_soft, target) * 10


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
