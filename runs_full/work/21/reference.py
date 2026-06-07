import torch
import torch.nn.functional as F
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: 256,8,36,64
        k: 256,8,36,64
        v: 256,8,36,64
        mask: 256,1,1,36
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        """
        mask(256,1,1,36)
        attn(256,8,36,36)
        这里用到了tensor的broadcast: 两个tensor必须满足，从最后一个维度开始往前算，维度要么相等，要么为1，要么不存在
        这里的mask中间两个维度为1，可以与attn做broadcast

        将mask的行索引复制到36，得到36×36的mask矩阵，batch中共256个36*36的矩阵，1/256即batch中每个样本的mask再复制到head=8个
        每个batch中样本的mask和各自的互注意力矩阵相乘
        注意力矩阵是36*36是个混淆矩阵，表示第一个元素和其余36个元素的关系，因此mask行列转置无所谓
        """
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
