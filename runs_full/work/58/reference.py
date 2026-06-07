import torch
import torch.onnx


class ToHalf(torch.nn.Module):

    def forward(self, tensor):
        return tensor.half()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
