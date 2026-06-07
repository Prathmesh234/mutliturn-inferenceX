import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, dims=(1, 2, 3)) ->None:
        super(DiceLoss, self).__init__()
        self.eps: 'float' = 1e-06
        self.dims = dims

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights=None) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.
                format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxNxHxW. Got: {}'
                .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(
                'input and target shapes must be the same. Got: {}'.format(
                input.shape))
        if not input.device == target.device:
            raise ValueError(
                'input and target must be in the same device. Got: {}'.
                format(input.device))
        smooth = 1
        input_soft = F.softmax(input, dim=1)
        intersection = torch.sum(input_soft * target, self.dims)
        cardinality = torch.sum(input_soft + target, self.dims)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth +
            self.eps)
        return torch.mean(1.0 - dice_score)


class MDiceLoss(nn.Module):

    def __init__(self) ->None:
        super(MDiceLoss, self).__init__()
        self.dice_loss_bg = DiceLoss(dims=0)
        self.dice_loss_fg = DiceLoss(dims=1)
        self.dice_loss_th = DiceLoss(dims=2)
        self.dice_loss_gp = DiceLoss(dims=3)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', w=None
        ) ->torch.Tensor:
        bg = self.dice_loss_bg(input, target, w)
        fg = self.dice_loss_fg(input, target, w)
        tg = self.dice_loss_th(input, target, w)
        gp = self.dice_loss_gp(input, target, w)
        return bg * 0.2 + fg * 1.1 + tg * 3 + gp * 3


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
