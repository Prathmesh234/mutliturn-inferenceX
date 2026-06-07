import math
import torch
import torch.nn.functional as F


class AttentionModuleV2(torch.nn.Module):

    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None,
        fc_spt_value=None, fc_x_update=None, fc_update=None,
        fc_spt_spt_query=None, fc_spt_spt_key=None, fc_spt_spt_value=None,
        gamma_scale_gate=None, gamma_bias_gate=None, beta_scale_gate=None):
        super().__init__()
        self.hidden_size = hidden_size
        if fc_x_query is not None:
            self.fc_x_query = fc_x_query
        else:
            self.fc_x_query = torch.nn.Linear(hidden_size, hidden_size,
                bias=False)
        if fc_spt_key is not None:
            self.fc_spt_key = fc_spt_key
        else:
            self.fc_spt_key = torch.nn.Linear(hidden_size, hidden_size,
                bias=False)
        if fc_spt_value is not None:
            self.fc_spt_value = fc_spt_value
        else:
            self.fc_spt_value = torch.nn.Linear(hidden_size, hidden_size,
                bias=False)
        if fc_x_update is not None:
            self.fc_x_update = fc_x_update
        else:
            self.fc_x_update = torch.nn.Linear(2 * hidden_size, hidden_size,
                bias=True)
        if fc_update is not None:
            self.fc_update = fc_update
        else:
            self.fc_update = torch.nn.Linear(2 * hidden_size, 2 *
                hidden_size, bias=True)
        if fc_spt_spt_query is not None:
            self.fc_spt_spt_query = fc_spt_spt_query
        else:
            self.fc_spt_spt_query = torch.nn.Linear(hidden_size,
                hidden_size, bias=False)
        if fc_spt_spt_key is not None:
            self.fc_spt_spt_key = fc_spt_spt_key
        else:
            self.fc_spt_spt_key = torch.nn.Linear(hidden_size, hidden_size,
                bias=False)
        if fc_spt_spt_value is not None:
            self.fc_spt_spt_value = fc_spt_spt_value
        else:
            self.fc_spt_spt_value = torch.nn.Linear(hidden_size,
                hidden_size, bias=False)
        if gamma_scale_gate is not None:
            self.gamma_scale_gate = gamma_scale_gate
        else:
            self.gamma_scale_gate = torch.nn.Parameter(torch.zeros(size=[1,
                hidden_size, 1, 1, 1], requires_grad=True))
        if gamma_bias_gate is not None:
            self.gamma_bias_gate = gamma_bias_gate
        else:
            self.gamma_bias_gate = torch.nn.Parameter(torch.ones(size=[1,
                hidden_size, 1, 1, 1], requires_grad=True))
        if beta_scale_gate is not None:
            self.beta_scale_gate = beta_scale_gate
        else:
            self.beta_scale_gate = torch.nn.Parameter(torch.zeros(size=[1,
                hidden_size, 1, 1, 1], requires_grad=True))

    def forward(self, x, proto_spt):
        proto_x = x.mean(axis=3).mean(axis=2)
        proto_x = proto_x.unsqueeze(dim=1)
        proto_spt = proto_spt.unsqueeze(dim=0)
        query = self.fc_x_query(proto_x)
        key = self.fc_spt_key(proto_spt)
        value = self.fc_spt_value(proto_spt)
        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        aggregated_messages = torch.matmul(correlation, value)
        proto_x = self.fc_x_update(torch.cat([proto_x, aggregated_messages],
            dim=-1))
        proto_spt = proto_spt + proto_x
        query = self.fc_spt_spt_query(proto_spt)
        key = self.fc_spt_spt_key(proto_spt)
        value = self.fc_spt_spt_value(proto_spt)
        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        proto_spt = torch.matmul(correlation, value)
        query = self.fc_x_query(proto_x)
        key = self.fc_spt_key(proto_spt)
        value = self.fc_spt_value(proto_spt)
        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        aggregated_messages = torch.matmul(correlation, value)
        film_params = self.fc_update(torch.cat([proto_x,
            aggregated_messages], dim=-1))
        gamma = film_params[:, 0, :self.hidden_size].unsqueeze(dim=2
            ).unsqueeze(dim=3).unsqueeze(dim=-1)
        beta = film_params[:, 0, self.hidden_size:].unsqueeze(-1).unsqueeze(-1
            ).unsqueeze(dim=-1)
        gamma = gamma * self.gamma_scale_gate + self.gamma_bias_gate
        beta = beta * self.beta_scale_gate
        x = gamma * x.unsqueeze(dim=-1) + beta
        x = x.squeeze(dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
