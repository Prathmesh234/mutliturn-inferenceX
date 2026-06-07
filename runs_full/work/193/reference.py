import torch
import torch.nn as nn
import torch.distributed
import torch.optim.lr_scheduler
import torch.utils.data


class HadamardProduct(nn.Module):

    def __init__(self, idim_1, idim_2, hdim):
        super(HadamardProduct, self).__init__()
        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, x1, x2):
        """
        Args:
            inp1: [B,idim_1] or [B,L,idim_1]
            inp2: [B,idim_2] or [B,L,idim_2]
        """
        return torch.relu(self.fc_3(torch.relu(self.fc_1(x1)) * torch.relu(
            self.fc_2(x2))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'idim_1': 4, 'idim_2': 4, 'hdim': 4}]
