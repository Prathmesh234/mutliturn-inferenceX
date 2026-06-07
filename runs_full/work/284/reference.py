import torch
import torch.nn as nn


def vflip(input: 'torch.Tensor') ->torch.Tensor:
    return torch.flip(input, [-2])


class Vflip(nn.Module):
    """Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The vertically flipped image tensor

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def __init__(self) ->None:
        super(Vflip, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return vflip(input)

    def __repr__(self):
        return self.__class__.__name__


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
