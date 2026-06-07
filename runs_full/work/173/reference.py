import torch
import torch.nn as nn
import torch.utils.checkpoint


def norm(x, axis=None, eps=1e-05):
    if axis is not None:
        return (x - x.mean(axis, True)) / (x.std(axis, keepdim=True) + eps)
    else:
        return (x - x.mean()) / (x.std() + eps)


class NormalSamples(nn.Module):
    """The [reparameterization trick](https://arxiv.org/abs/1312.6114v10) for sampling values from Gaussian distributions with learned mean & stddev.

    The input vector must be twice as big as the output. And, normalize it, since we don't impose a loss-based regularization on mean & stddev here like VAEs do."""

    def __init__(self):
        super().__init__()

    def forward(self, mean_std):
        mean, std = mean_std.split(int(mean_std.shape[-1]) // 2, -1)
        mean, std = norm(mean), norm(std) + 1
        noise = torch.randn_like(mean)
        return mean + noise


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
