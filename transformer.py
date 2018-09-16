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
   #     input_norm = self.layer_norm(inputs)
        input_norm_split = torch.split(inputs, 512, dim = 2)
   #     value_norm = self.layer_norm(values)
        value_norm_split = torch.split(values, 512, dim = 2)
        for value_norm_split1 in value_norm_split:
            print('value b4 cat in lattice encoder in transformer', value_norm_split1.size())#test
        value_norm_split = torch.cat(value_norm_split, dim = 1)

        print('value_norm_split type', type(value_norm_split))
        for input_norm_split1 in input_norm_split:
            print('input in lattice encoder in transformer', input_norm_split1.size())#test
            input_norm_split2 = input_norm_split1
        query_valkey_test = zip(input_norm_split, value_norm_split)
        print('value in lattice encoder in transformer', value_norm_split.size())#test
      #  for value_norm_split1 in value_norm_split:
       #     print('value  individual in lattice encoder in transformer', value_norm_split1.size())#test
        for input_norm_split1, value_norm_split1 in query_valkey_test:
            print('input, value in lattice encoder in transformer', input_norm_split1.size(), value_norm_split.size())#test
        query_valkey = zip(input_norm_split, value_norm_split)
     #   context = torch.cat([self.latt_attn(input_norm_split2, value_norm_split2) for
         #                         input_norm_split2, value_norm_split2 in query_valkey], dim = 1)
        context, _ = self.latt_attn(input_norm_split2, value_norm_split)
        print('type of context inputs in transformer', type(context), type(inputs))
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
                 dropout, embeddings, embeddings_latt = False):  
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
#latt
        self.embeddings_latt = embeddings_latt
#latt
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
#latt
        self.lattice = nn.ModuleList(
            [LatticeEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(3)])
#latt
        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, src, lengths=None, feat_merge = False):
                                                    #latt
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        print('src in transformer encoder', src.size())#test
        print('embed in transformer encoder ', self.embeddings) #test

#latt
# check if embeddings of features is available
# if so, obtain embeddings of word and senses separately
        print('feat_merge in transformer', feat_merge)
        if feat_merge == 'latt':
            emb, emb_latt = self.embeddings(src)
        else:
            emb = self.embeddings(src)
#latt

        print('emb type in transformer encoder', type(emb)) #test
        print('emb in transformer encoder', emb.size()) #test
        out = emb.transpose(0, 1).contiguous()
        print('out in transformer encoder', out.size()) #test
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
 #           print(i, 'out', out) #test
            print('size before normal transformer', out.size(), mask.size())#test
            out = self.transformer[i](out, mask)


#latt
#  if lattice is used, run lattice layers after normal transformer encoding
        out_latt = emb_latt.transpose(0, 1).contiguous()
        if feat_merge == 'latt':
            for i in range(3):
                print('size before lattice', out.size(), out_latt.size())#test
                out = self.lattice[i](out, out_latt)
#latt

        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous()
