import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=3, padding=1)
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=out_channels,
            momentum=1.0, track_running_stats=False)
        self.cached_support_features = None

    def forward(self, x, is_support=False):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if is_support:
            self.cached_support_features = x.detach()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
