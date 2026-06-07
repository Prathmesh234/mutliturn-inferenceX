import torch
import torch.nn as nn


def bgr_to_rgb(image: 'torch.Tensor') ->torch.Tensor:
    """Convert a BGR image to RGB.

    Args:
        image (torch.Tensor): BGR Image to be converted to BGR of shape :math:`(*,3,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W).Got {}'
            .format(image.shape))
    out: 'torch.Tensor' = image.flip(-3)
    return out


def rgb_to_bgr(image: 'torch.Tensor') ->torch.Tensor:
    """Convert a RGB image to BGR.

    Args:
        image (torch.Tensor): RGB Image to be converted to BGRof of shape :math:`(*,3,H,W)`.

    Returns:
        torch.Tensor: BGR version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W).Got {}'
            .format(image.shape))
    return bgr_to_rgb(image)


def rgba_to_rgb(image: 'torch.Tensor') ->torch.Tensor:
    """Convert an image from RGBA to RGB.

    Args:
        image (torch.Tensor): RGBA Image to be converted to RGB of shape :math:`(*,4,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(
            f'Input size must have a shape of (*, 4, H, W).Got {image.shape}')
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)
    a_one = torch.tensor(1.0) - a
    a_one * r + a * r
    a_one * g + a * g
    a_one * b + a * b
    return torch.cat([r, g, b], dim=-3)


def rgba_to_bgr(image: 'torch.Tensor') ->torch.Tensor:
    """Convert an image from RGBA to BGR.

    Args:
        image (torch.Tensor): RGBA Image to be converted to BGR of shape :math:`(*,4,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_bgr(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(
            f'Input size must have a shape of (*, 4, H, W).Got {image.shape}')
    x_rgb: 'torch.Tensor' = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


class RgbaToBgr(nn.Module):
    """Convert an image from RGBA to BGR.

    Remove an alpha channel from BGR image.

    Returns:
        torch.Tensor: BGR version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def __init__(self) ->None:
        super(RgbaToBgr, self).__init__()

    def forward(self, image: 'torch.Tensor') ->torch.Tensor:
        return rgba_to_bgr(image)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
