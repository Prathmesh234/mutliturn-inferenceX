import torch
import torch.nn as nn


class DenseNet2D_up_block_concat(nn.Module):

    def __init__(self, skip_channels, input_channels, output_channels,
        up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels,
            output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels,
            kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(skip_channels + input_channels +
            output_channels, output_channels, kernel_size=(1, 1), padding=(
            0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels,
            kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x):
        x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode=
            'nearest')
        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        """
        deltaTime1 = time1 - time0
        deltaTime2 = time2 - time1
        deltaTime3 = time3 - time2
        deltaTime4 = time4 - time3
        deltaTime5 = time5 - time4

        print("UpBlock " + str(deltaTime1) + ' ' + str(deltaTime2) + ' ' + str(deltaTime3) + ' ' + str(deltaTime4) + ' ' + str(deltaTime5))
        """
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'skip_channels': 4, 'input_channels': 4, 'output_channels':
        4, 'up_stride': 1}]
