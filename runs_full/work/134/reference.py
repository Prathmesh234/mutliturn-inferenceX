import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    """
    Implementation of Attentive Pooling 
    """

    def __init__(self, input_dim, **kwargs):
        super(AttentivePooling, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = nn.ReLU()
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


class ASP(nn.Module):
    """ Attentive Statistic Pooling module incoporate attention mask"""

    def __init__(self, out_dim, input_dim):
        super(ASP, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        self.ap_layer = AttentivePooling(out_dim)

    def forward(self, feature_BxTxH, att_mask_BxT):
        """ 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        """
        feature_BxTxH = self.linear(feature_BxTxH)
        sap_vec, att_w = self.ap_layer(feature_BxTxH, att_mask_BxT)
        variance = torch.sqrt(torch.sum(att_w * feature_BxTxH *
            feature_BxTxH, dim=1) - sap_vec ** 2 + 1e-08)
        statistic_pooling = torch.cat([sap_vec, variance], dim=-1)
        return statistic_pooling


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_dim': 4, 'input_dim': 4}]
