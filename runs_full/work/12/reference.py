import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class SimpleBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if
            half_res else 1, padding=1, bias=False)
        self.BN1 = nn.Identity()
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=
            False)
        self.BN2 = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 
                1, bias=False)
            self.BNshortcut = nn.Identity()
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(
            self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 2, 2])]


def get_init_inputs():
    return [[], {'indim': 4, 'outdim': 4, 'half_res': 4}]
