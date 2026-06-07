import torch
from torch.autograd import Variable
import triton
import triton.language as tl


@triton.jit
def _mse_kernel(x_ptr, y_ptr, out_ptr, n_elements, inv_n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    s = tl.sum(d * d, axis=0)
    tl.store(out_ptr, s * inv_n)


def torch_norm_quat(quat, USE_CUDA=True):
    batch_size = quat.size()[0]
    quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    for i in range(batch_size):
        norm_quat = torch.norm(quat[i])
        if norm_quat > 1e-06:
            quat_out[i] = quat[i] / norm_quat
        else:
            quat_out[i, :3] = quat[i, :3] * 0
            quat_out[i, 3] = quat[i, 3] / quat[i, 3]
    return quat_out


def torch_QuaternionProduct(q1, q2, USE_CUDA=True):
    x1 = q1[:, 0]; y1 = q1[:, 1]; z1 = q1[:, 2]; w1 = q1[:, 3]
    x2 = q2[:, 0]; y2 = q2[:, 1]; z2 = q2[:, 2]; w2 = q2[:, 3]
    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    quat[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    quat[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    quat[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    quat[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    quat = torch_norm_quat(quat)
    return quat


class Follow_lossNew(torch.nn.Module):

    def __init__(self):
        super(Follow_lossNew, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, virtual_quat, real_quat, real_postion=None):
        if real_postion is not None:
            real_quat = torch_QuaternionProduct(real_quat, real_postion)
        x = virtual_quat.contiguous()
        y = real_quat.contiguous()
        n = x.numel()
        out = torch.empty([], device=x.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _mse_kernel[(1,)](x, y, out, n, 1.0 / n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out
