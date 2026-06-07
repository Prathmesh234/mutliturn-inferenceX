import torch
from torch import nn
import torch.optim.lr_scheduler


class Classifier(nn.Module):

    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_size': 4, 'output_size': 4}]
