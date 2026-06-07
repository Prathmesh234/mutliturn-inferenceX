import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, 128, kernel_size=4, stride
            =1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
            padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2,
            padding=1, bias=False)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.lrelu(x)
        x = self.deconv2(x)
        x = self.lrelu(x)
        x = self.deconv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4}]
