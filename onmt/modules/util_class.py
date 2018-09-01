""" Misc classes """
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        print(self.a_2.size(), x.size())
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp', 'latt', 'latt_senses'] #latt
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        print(len(self), len(inputs_)) #test
#latt
        if self.merge == 'latt':
            print('len self', len(self)) #test
            assert len(self) == 1
        elif self.merge == 'concat':
            print('len self', len(self)) #test
            assert len(self) == 1
        elif self.merge == 'mlp':
            print('len self', len(self)) #test
            assert len(self) == 1
        elif self.merge == 'latt_senses':
            print('len self', len(self)) #test
            assert len(self) == 1
        elif self.merge == 'sum':
            print('len self', len(self)) #test
            assert len(self) == 1
        else:
            print('len self', len(self), len(inputs_)) #test
            assert len(self) == len(inputs_)
#latt
        outputs = [f(x) for f, x in zip(self, inputs_)]
        print('outputs in util_class.py', outputs[0].size(), outputs)
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        elif self.merge == 'latt_senses':
            return torch.cat(torch.unbind(outputs[0]))
        else:
            return outputs[0] #latt solve problem of list, otherwise, error pops up!
