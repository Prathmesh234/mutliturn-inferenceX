import math
import torch
import torch.nn as nn
from numpy import prod


def getLayerNormalizationFactor(x):
    """
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    """
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self, module, equalized=True, lrMul=1.0, initBiasToZero=True):
        """
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """
        super(ConstrainedLayer, self).__init__()
        self.module = module
        self.equalized = equalized
        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedLinear(ConstrainedLayer):

    def __init__(self, nChannelsPrevious, nChannels, bias=True, **kwargs):
        """
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self, nn.Linear(nChannelsPrevious,
            nChannels, bias=bias), **kwargs)


class AdaIN(nn.Module):

    def __init__(self, dimIn, dimOut, epsilon=1e-08):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = EqualizedLinear(dimIn, 2 * dimOut, equalized=
            True, initBiasToZero=True)
        self.dimOut = dimOut

    def forward(self, x, y):
        batchSize, nChannel, _width, _height = x.size()
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        varx = torch.clamp((tmpX * tmpX).mean(dim=2).view(batchSize,
            nChannel, 1, 1) - mux * mux, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - mux) * varx
        styleY = self.styleModulator(y)
        yA = styleY[:, :self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)
        return yA * x + yB


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dimIn': 4, 'dimOut': 4}]
