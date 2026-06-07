import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)
        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)
        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)
        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)
        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBottleNeck(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        super(ShakeBottleNeck, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = None if self.equal_io else Shortcut(in_ch, out_ch,
            stride=stride)
        self.branch1 = self._make_branch(in_ch, mid_ch, out_ch, cardinary,
            stride)
        self.branch2 = self._make_branch(in_ch, mid_ch, out_ch, cardinary,
            stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        return nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, padding=0, bias=
            False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False), nn.
            Conv2d(mid_ch, mid_ch, 3, padding=1, stride=stride, groups=
            cardinary, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace
            =False), nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNeXt(nn.Module):

    def __init__(self, depth, w_base, cardinary, label):
        super(ShakeResNeXt, self).__init__()
        n_units = (depth - 2) // 9
        n_chs = [64, 128, 256, 1024]
        self.n_chs = n_chs
        self.in_ch = n_chs[0]
        self.c_in = nn.Conv2d(3, n_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, n_chs[0], w_base, cardinary)
        self.layer2 = self._make_layer(n_units, n_chs[1], w_base, cardinary, 2)
        self.layer3 = self._make_layer(n_units, n_chs[2], w_base, cardinary, 2)
        self.fc_out = nn.Linear(n_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.n_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, n_ch, w_base, cardinary, stride=1):
        layers = []
        mid_ch, out_ch = n_ch * (w_base // 64) * cardinary, n_ch * 4
        for i in range(n_units):
            layers.append(ShakeBottleNeck(self.in_ch, mid_ch, out_ch,
                cardinary, stride=stride))
            self.in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'depth': 1, 'w_base': 4, 'cardinary': 4, 'label': 4}]
