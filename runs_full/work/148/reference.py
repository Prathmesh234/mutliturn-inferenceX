import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import functional as F


class ActorCriticMLP(nn.Module):
    """MLP network with heads for actor and critic."""

    def __init__(self, input_shape: 'Tuple[int]', n_actions: 'int',
        hidden_size: 'int'=128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden_size)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x) ->Tuple[Tensor, Tensor]:
        """Forward pass through network. Calculates the action logits and the value.

        Args:
            x: input to network

        Returns:
            action log probs (logits), value
        """
        x = F.relu(self.fc1(x.float()))
        a = F.log_softmax(self.actor_head(x), dim=-1)
        c = self.critic_head(x)
        return a, c


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'n_actions': 4}]
