import torch
import torch.nn as tnn


class Net(tnn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = tnn.Conv2d(3, 6, 5)
        self.pool = tnn.MaxPool2d(2, 2)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.fc1 = tnn.Linear(16 * 5 * 5, 120)
        self.fc2 = tnn.Linear(120, 84)
        self.fc3 = tnn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]
