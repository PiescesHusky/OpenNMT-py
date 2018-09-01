"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        print('size input norm', input_norm.size())
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

#latt
class LatticeEncoderLayer(nn.Module):
    """
    A single layer of the lattice encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(LatticeEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
  #latt
        self.latt_attn = onmt.modules.GlobalAttention(d_model)
  #latt
        self.layer_norm = onmt.modules.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, values):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        input_norm_split = torch.split(input_norm, 1, dim = 1)
        value_norm = self.layer_norm(values)
        value_norm_split = torch.split(value_norm, 5, dim = 1)
        query_valkey = zip(input_norm_split, value_norm_split)
        context = torch.cat([self.latt_attn(input_norm_split, value_norm_split) for
                                  input_norm_split, value_norm_split in query_valkey], dim = 1)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)
#latt


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
#latt
        self.lattice = nn.ModuleList(
            [LatticeEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(3)])
#latt
        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        print('src', src.size())#test
        print('embed', self.embeddings) #test
        emb = self.embeddings(src)
        print('emb', emb.size()) #test
        out = emb.transpose(0, 1).contiguous()
        print('out', out.size()) #test
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
 #           print(i, 'out', out) #test
            print('size', out.size(), mask.size())#test
            out = self.transformer[i](out, mask)
#latt
     #   for i in range(3):
        #    out = self.lattice[i](out, multi_senses)
#latt
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous()
