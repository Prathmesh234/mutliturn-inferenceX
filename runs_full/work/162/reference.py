import torch
import numpy as np
from torch import nn


class Rosenbrock(nn.Module):

    def __init__(self, n1, n2, a=1.0 / 20.0, b=5.0):
        super(Rosenbrock, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.a = a
        self.b = b

    def forward(self, x):
        dim2 = x.ndimension() > 2
        dim1 = x.ndimension() > 1
        if dim2:
            y = x[:, :, 0]
            x = torch.reshape(x[:, :, 1:], (x.size()[0], x.size()[1], self.
                n2, self.n1 - 1))
            xx = x[:, :, :, 1:]
            xxx = x[:, :, :, 0:-1]
            result = -self.a * (y - 1) ** 2
            -self.b * torch.sum(torch.sum((xx - xxx ** 2) ** 2, -1), -1)
        else:
            x = x if dim1 else x.unsqueeze(0)
            y = x[:, 0]
            x = torch.reshape(x[:, 1:], (x.size()[0], self.n2, self.n1 - 1))
            xx = x[:, :, 1:]
            xxx = x[:, :, 0:-1]
            result = -self.a * (y - 1) ** 2 - self.b * torch.sum(torch.sum(
                (xx - xxx ** 2) ** 2, -1), -1)
        return result if dim1 else result.squeeze(0)

    def normalization(self):
        return (1 / 20) ** (1 / 2) * 5 ** (self.n2 * (self.n1 - 1) / 2
            ) / np.pi ** ((self.n2 * (self.n1 - 1) + 1) / 2)

    def Iid(self, N):
        a = self.a
        b = self.b
        mu = 1
        S = np.zeros((1, self.n2 * (self.n1 - 1) + 1))
        for k in range(N):
            s = np.array([[]])
            y = np.random.normal(mu, 1 / (2 * a), size=(1, 1))
            s = np.concatenate((s, y), 1)
            for j in range(1, self.n2 + 1):
                z = y
                for i in range(2, self.n1 + 1):
                    x = np.random.normal(z ** 2, 1 / (2 * b), size=(1, 1))
                    s = np.concatenate((s, x), 1)
                    z = x
            S = np.concatenate((S, s))
        return S[1:, :]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n1': 4, 'n2': 4}]
