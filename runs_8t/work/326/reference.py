import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedVectorAttention(nn.Module):
    """vector attention"""

    def __init__(self, input_dim, hidden_dim):
        super(BatchedVectorAttention, self).__init__()
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(input_dim, hidden_dim)
        self.psi = nn.Linear(input_dim, hidden_dim)
        self.recover1 = nn.Linear(hidden_dim, max(input_dim // 2, 1))
        self.lrelu = nn.LeakyReLU(0.2)
        self.recover2 = nn.Linear(max(input_dim // 2, 1), input_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: [n, L, c]
        """
        n, L, c = x.shape
        x.view(-1, c)
        x_t = self.theta(x).view(n, L, -1)
        x_ph = self.phi(x).view(n, L, -1)
        x_psi = self.psi(x).view(n, L, -1)
        attention_map = torch.matmul(x_ph, torch.transpose(x_t, 1, 2))
        attention_map = attention_map
        attention_map = F.softmax(attention_map, dim=2)
        x_add = torch.matmul(attention_map, x_psi)
        x_add = self.recover1(x_add.view(n * L, -1))
        x_add = self.lrelu(x_add)
        x_add = self.recover2(x_add)
        x_add = self.tanh(x_add)
        x_add = x_add.view(n, L, c)
        return x + x_add


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
