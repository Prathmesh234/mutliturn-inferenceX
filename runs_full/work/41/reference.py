import torch


class NormActivation(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        tensor = tensor ** 2
        length = tensor.sum(dim=self.dim, keepdim=True)
        return tensor / length


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
