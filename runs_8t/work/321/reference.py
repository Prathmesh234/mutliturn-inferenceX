import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.onnx


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3 * 3 * 64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 3 * 3 * 64)
        x = self.fc1(x)
        return F.log_softmax(x)


def get_inputs():
    return [torch.rand([4, 1, 32, 32])]


def get_init_inputs():
    return [[], {}]
