import torch
from typing import Tuple
from typing import Optional


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0):
        """Processes a projected query and key-value pair to apply
        scaled dot product attention.
        Args:
            dropout (float): probability of dropping an attention weight.
        Examples::
            >>> SDP = torchtext.models.ScaledDotProduct(0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value:
        'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, bias_k:
        'Optional[torch.Tensor]'=None, bias_v: 'Optional[torch.Tensor]'=None
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """Uses a scaled dot product with the projected key-value pair to update
        the projected query.
        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k and bias_v: (Tensor, optional): one more key and value sequence to be added at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                non-None to both arguments in order to activate them.
        Shape:
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`
            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2
                ) == bias_k.size(-2) and bias_k.size(-3
                ) == 1, 'Shape of bias_k is not supported'
            assert value.size(-1) == bias_v.size(-1) and value.size(-2
                ) == bias_v.size(-2) and bias_v.size(-3
                ) == 1, 'Shape of bias_v is not supported'
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                _attn_mask = attn_mask
                attn_mask = torch.nn.functional.pad(_attn_mask, (0, 1))
        tgt_len, head_dim = query.size(-3), query.size(-1)
        assert query.size(-1) == key.size(-1) == value.size(-1
            ), 'The feature dim of query, key, value must be equal.'
        assert key.size() == value.size(), 'Shape of key, value must match'
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3
            ), value.transpose(-2, -3)
        query = query * head_dim ** -0.5
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if attn_mask.size(-1) != src_len or attn_mask.size(-2
                ) != tgt_len or attn_mask.size(-3) != 1 and attn_mask.size(-3
                ) != batch_heads:
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError(
                    'Only bool tensor is supported for attn_mask')
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -100000000.0)
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights,
            dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights,
            p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return attn_output.transpose(-2, -3), attn_output_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
