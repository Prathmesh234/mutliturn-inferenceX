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


class BayesLinear(nn.Module):
    """
    This class implements a Bayesian Linear layer, which has a distribution instead of weights. 
    """

    def __init__(self, in_features, out_features, bias=True,
        log_sigma_prior=-5, mu_prior=-1):
        """
        Initializes a BayesLinear layer. 

        Args:
            in_features: number of input features
            out_features: number of output features
            bias: whether to add bias  
            log_sigma_prior: the initial value of the standard deviation of the distribution
            mu_prior: the initial value of the mean of the distribution
        """
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features)
            )
        self.mu_prior_init = mu_prior
        self.log_sigma_prior_init = log_sigma_prior
        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_features))
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
        act_mu = F.linear(input, self.w_mu, self.bias)
        act_sigma = torch.sqrt(F.linear(input ** 2, torch.exp(self.
            w_log_sigma) ** 2) + 1e-08)
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
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
