# -*- coding: utf-8 -*-
"Composite sub modules used in VAE networks."

import torch
import torch.nn as nn

import torch.distributions as tdist


def create_linear_layer(
        in_features,
        out_features,
        bn_args=dict(),
        dropout_rate=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, **bn_args),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None)


class LinearLayers(nn.Module):
    """Convenience module for creating multiple layers with the same activation function."""

    def __init__(self, dims, dropout_rate=0):
        """Initialize LinearLayers object.

        Args:
            dims (List): List with layer dimensions.
        """
        super().__init__()

        self.layers = nn.ModuleList([create_linear_layer(dims[i-1], dims[i]) for i in range(1, len(dims))])
        self.dims = dims


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            assert not torch.isnan(x).any(), 'Layer ' + str(i) + ' output is nan!'
            x = layer(x)
        return x


def create_stochastic_net(d_input, d_hidden, d_output, stochastic_layer):
    """Creates a neural net with stochastic output layer."""

    if d_hidden is None:
        net = torch.nn.modules.Identity() 
        sl = stochastic_layer(d_input, d_output)
    else:
        net = LinearLayers([d_input, *d_hidden])
        sl = stochastic_layer(d_hidden[-1], d_output)
    return nn.Sequential(net, sl)
