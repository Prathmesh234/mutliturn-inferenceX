import torch
from torch.autograd import Variable


def torch_norm_quat(quat, USE_CUDA=True):
    batch_size = quat.size()[0]
    quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA is True:
        quat_out = quat_out
    for i in range(batch_size):
        norm_quat = torch.norm(quat[i])
        if norm_quat > 1e-06:
            quat_out[i] = quat[i] / norm_quat
        else:
            quat_out[i, :3] = quat[i, :3] * 0
            quat_out[i, 3] = quat[i, 3] / quat[i, 3]
    return quat_out


def torch_QuaternionProduct(q1, q2, USE_CUDA=True):
    x1 = q1[:, 0]
    y1 = q1[:, 1]
    z1 = q1[:, 2]
    w1 = q1[:, 3]
    x2 = q2[:, 0]
    y2 = q2[:, 1]
    z2 = q2[:, 2]
    w2 = q2[:, 3]
    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA is True:
        quat = quat
    quat[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    quat[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    quat[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    quat[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    quat = torch_norm_quat(quat)
    return quat


class Follow_loss(torch.nn.Module):

    def __init__(self):
        super(Follow_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, virtual_quat, real_quat, real_postion=None):
        if real_postion is not None:
            real_quat = torch_QuaternionProduct(real_quat, real_postion)
        return self.MSE(virtual_quat, real_quat)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
