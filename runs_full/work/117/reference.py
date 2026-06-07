import torch
import torch.nn as nn


class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """

    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep, att_w


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
