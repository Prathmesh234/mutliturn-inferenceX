import torch
import torch.nn as nn


class PLU(nn.Module):
    """
    y = max(alpha*(x+c)−c, min(alpha*(x−c)+c, x))
    from PLU: The Piecewise Linear Unit Activation Function
    """

    def __init__(self, alpha=0.1, c=1):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, x):
        x1 = self.alpha * (x + self.c) - self.c
        x2 = self.alpha * (x - self.c) + self.c
        min1 = torch.min(x2, x)
        min2 = torch.max(x1, min1)
        return min2

    def __repr__(self):
        s = '{name} ({alhpa}, {c})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
