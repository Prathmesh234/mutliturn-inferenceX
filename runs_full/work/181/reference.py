import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, node_dim, hid_dim, num_class_l1, num_class_l2,
        num_class_l3):
        super(Linear, self).__init__()
        self.linear_l1 = nn.Linear(node_dim, num_class_l1)
        self.linear_l2 = nn.Linear(node_dim + num_class_l1, num_class_l2)
        self.linear_l3 = nn.Linear(node_dim + num_class_l2, num_class_l3)

    def forward(self, x, y1, y2):
        yp_l1 = self.linear_l1(x)
        yp_l2 = self.linear_l2(torch.cat((x, y1), dim=-1))
        yp_l3 = self.linear_l3(torch.cat((x, y2), dim=-1))
        return yp_l1, yp_l2, yp_l3

    @torch.no_grad()
    def predict(self, x):
        yp_l1 = F.softmax(self.linear_l1(x), dim=-1)
        yp_l2 = F.softmax(self.linear_l2(torch.cat((x, yp_l1), dim=-1)), dim=-1
            )
        yp_l3 = F.softmax(self.linear_l3(torch.cat((x, yp_l2), dim=-1)), dim=-1
            )
        return yp_l1, yp_l2, yp_l3


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'node_dim': 4, 'hid_dim': 4, 'num_class_l1': 4,
        'num_class_l2': 4, 'num_class_l3': 4}]
