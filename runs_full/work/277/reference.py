import torch
import torch.nn as nn


def hflip(input: 'torch.Tensor') ->torch.Tensor:
    return torch.flip(input, [-1])


class Hflip(nn.Module):
    """Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])
    """

    def __init__(self) ->None:
        super(Hflip, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return hflip(input)

    def __repr__(self):
        return self.__class__.__name__


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
