import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class GeometryFeature(nn.Module):

    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 
        4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
