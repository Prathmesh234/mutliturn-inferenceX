import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class SDNE_layer(nn.Module):

    def __init__(self, num_node, hidden_size1, hidden_size2, droput, alpha,
        beta, nu1, nu2):
        super(SDNE_layer, self).__init__()
        self.num_node = num_node
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.droput = droput
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.encode0 = nn.Linear(self.num_node, self.hidden_size1)
        self.encode1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.decode0 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.decode1 = nn.Linear(self.hidden_size1, self.num_node)

    def forward(self, adj_mat, l_mat):
        t0 = F.leaky_relu(self.encode0(adj_mat))
        t0 = F.leaky_relu(self.encode1(t0))
        self.embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        L_1st = 2 * torch.trace(torch.mm(torch.mm(torch.t(self.embedding),
            l_mat), self.embedding))
        L_2nd = torch.sum((adj_mat - t0) * adj_mat * self.beta * ((adj_mat -
            t0) * adj_mat * self.beta))
        L_reg = 0
        for param in self.parameters():
            L_reg += self.nu1 * torch.sum(torch.abs(param)
                ) + self.nu2 * torch.sum(param * param)
        return self.alpha * L_1st, L_2nd, self.alpha * L_1st + L_2nd, L_reg

    def get_emb(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_node': 4, 'hidden_size1': 4, 'hidden_size2': 4,
        'droput': 4, 'alpha': 4, 'beta': 4, 'nu1': 4, 'nu2': 4}]
