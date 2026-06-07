import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    """
    Calculates the Kullback-Leibler divergence between two univariate Gaussians (p and q)

    Args:
        mu_p: mean of the Gaussian p
        sig_p: standard deviation of the Gaussian p
        mu_q: mean of the Gaussian q
        sig_q: standard deviation of the Gaussian q
    """
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) +
        ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


class BayesConv1d(nn.Module):
    """
    This class implements a Bayesian 1-dimensional Convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, bias=True, log_sigma_prior=-5, mu_prior=-1):
        """
        Initializes BayesConv1d layer. 

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of the convolutional kernel
            stride: stride of the convolution
            dilation: spacing between the kernel points of the convolution
            bias: whether to add bias  
            log_sigma_prior: the initial value of the standard deviation of the distribution
            mu_prior: the initial value of the mean of the distribution
           """
        super(BayesConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.w_mu = nn.Parameter(torch.Tensor(out_channels, in_channels,
            kernel_size))
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_channels,
            in_channels, kernel_size))
        self.mu_prior_init = mu_prior
        self.log_sigma_prior_init = log_sigma_prior
        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the layer
        """
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init - 0.1,
            self.log_sigma_prior_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Performs a forward pass of the input. Uses the Reparemetrization trick proposed by Kingma et al. 
        in "Variational Dropout and the Local Reparameterization trick" to sample directly from the activations.
        
        Args:
            input: the input to be forwarded
        """
        act_mu = F.conv1d(input, self.w_mu, self.bias, self.stride, self.
            padding, self.dilation)
        act_sigma = torch.sqrt(torch.clamp(F.conv1d(input ** 2, torch.exp(
            self.w_log_sigma) ** 2, self.bias, self.stride, self.padding,
            self.dilation), min=1e-16))
        epsilon = torch.randn_like(act_mu)
        return act_mu + act_sigma * epsilon

    def kl(self):
        """
        Returns the Kullback-Leibler divergence between the prior and the posterior of Bayesian layer.
        """
        return calculate_kl(torch.Tensor([self.mu_prior_init]), torch.exp(
            torch.Tensor([self.log_sigma_prior_init])), self.w_mu, torch.
            exp(self.w_log_sigma))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4, 'dilation': 1}]
