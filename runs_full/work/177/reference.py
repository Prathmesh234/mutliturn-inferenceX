import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast as autocast
import torch._C
import torch.serialization


class PPMConcat(nn.ModuleList):
    """Pyramid Pooling Module that only concat the features of each layer.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
    """

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__([nn.AdaptiveAvgPool2d(pool_scale) for
            pool_scale in pool_scales])

    def forward(self, feats):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = torch.cat(ppm_outs, dim=2)
        return concat_outs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
