import torch
from torch import nn
from torch.nn import functional as F


class Joiner(nn.Module):

    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_latent_dim + y_latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x_y = torch.cat([x, y], 1)
        x_y = F.relu(self.fc1(x_y))
        return self.fc2(x_y)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'x_latent_dim': 4, 'y_latent_dim': 4, 'hidden_dim': 4}]
