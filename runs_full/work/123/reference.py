import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (N, T, 1)
        
        return:
        utter_rep: size (N, H)
        """
        batch_rep.shape[1]
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep


class SAP(nn.Module):
    """ Self Attention Pooling module incoporate attention mask"""

    def __init__(self, out_dim):
        super(SAP, self).__init__()
        self.act_fn = nn.Tanh()
        self.sap_layer = SelfAttentionPooling(out_dim)

    def forward(self, feature, att_mask):
        """ 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        """
        feature = self.act_fn(feature)
        sap_vec = self.sap_layer(feature, att_mask)
        return sap_vec


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_dim': 4}]
