# -*- coding: utf-8 -*-
"Probabilistic neural network layers."

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from tempo.distributions.intervals import IntervalDistribution


class StochasticLayer(nn.Module):
    """
    Base stochastic layer.

    Stochastic layer which uses a single layer to map in_features to parameters
    of a distribution in out_features-dimensional space. Forward should return
    sublcass of torch.distributions.Distribution.
    """

    def __init__(self, in_features, out_features):
        """Initialize stochastic layer.

        Args:
            in_features: (int) Input dimensions.
            out_features: (int) Output dimensions.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features


class TorchDistributionSL(StochasticLayer):
    """
    Stochastic layer producing an i.i.d distribution.

    Should work with most univariate torch.distribution.Distribution
    classes.
    """

    def __init__(self, in_features, out_features, torch_dist, param_list, constant_buffers={}, constant_parameters={}):
        """Initialize stochastic layer.

        Args:
            in_features: (int) Input dimensions.
            out_features: (int) Output dimensions.
            torch_dist: (str) Name of distribution object to use.
            param_list: (list) List of strings of distribution parameters.
                This is necessary, as some distributions may be parameterized
                in more than one way.
            constant_buffers: (dict) dict of constant buffers.
            constant_parameters: (dict) dict of constant parameters.
        """
        super().__init__(in_features, out_features)

        self.param_list = param_list
        self.param_layers = nn.ModuleDict()
        self.torch_dist = torch_dist
        self.constraints = dict()

        self.constant_buffers = list(constant_buffers.keys())
        self.constant_parameters = list(constant_parameters.keys())

        for key, value in constant_buffers.items():
            self.register_buffer(key, value)

        for key, value in constant_parameters.items():
            self.register_parameter(key, value)

        for p in param_list:
            self.param_layers[p] = nn.Linear(self.in_features, self.out_features)
            self.constraints[p] = torch_dist.arg_constraints[p]


    def forward(self, x, *args):
        kwargs = dict()
        for p in self.param_layers.keys():
            transform = tdist.constraint_registry.transform_to(self.constraints[p])
            kwargs[p] = transform(self.param_layers[p](x))
        for p in self.constant_buffers:
            kwargs[p] = getattr(self, p)
        for p in self.constant_parameters:
            kwargs[p] = getattr(self, p)
        return tdist.Independent(self.torch_dist(**kwargs), 1)


NormalSL = partial(TorchDistributionSL, torch_dist=tdist.Normal, param_list=['loc', 'scale'])
LogNormalSL = partial(TorchDistributionSL, torch_dist=tdist.LogNormal, param_list=['loc', 'scale'])

BernoulliSL = partial(TorchDistributionSL, torch_dist=tdist.Bernoulli, param_list=['logits'])
PoissonSL = partial(TorchDistributionSL, torch_dist=tdist.Poisson, param_list=['rate'])
NegativeBinomialSL = partial(TorchDistributionSL, torch_dist=tdist.NegativeBinomial, param_list=['total_count', 'logits'])


class IntervalsSL(TorchDistributionSL):
    """
    Likelihood for intervals on the real axis.
    """

    def __init__(self, in_features, torch_dist, param_list, constant_buffers={}, constant_parameters={}):
        """Initialize stochastic layer.

        Args:
            in_features: (int) Input dimensions.
            torch_dist: (str) Name of distribution object to use.
            param_list: (list) List of strings of distribution parameters.
                This is necessary, as some distributions may be parameterized
                in more than one way.
            constant_buffers: (dict) dict of constant buffers.
            constant_parameters: (dict) dict of constant parameters.
        """
        super().__init__(in_features, 1, torch_dist, param_list, constant_buffers, constant_parameters)


    def forward(self, x):
        return IntervalDistribution(super().forward(x))


SingleScaleNormalIntervalsSL = partial(IntervalsSL, torch_dist=tdist.Normal, param_list=['loc'], constant_parameters={'scale' : torch.nn.Parameter(torch.Tensor([1.]))})
FixedScaleNormalIntervalsSL = partial(IntervalsSL, torch_dist=tdist.Normal, param_list=['loc'], constant_buffers={'scale' : torch.Tensor([1.])})


class ScaledBernoulliSL(StochasticLayer):
    """
    Bernoulli layer with size factor.
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

        self.p_mean = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.Softmax(dim=-1))


    def forward(self, x, l, y=None):
        p_mean = self.p_mean(x)
        p_mean = 1 - torch.exp(l * torch.log(1 - p_mean))
        return tdist.Independent(tdist.Bernoulli(probs=p_mean), 1)


class ScaledNegativeBinomialSL(StochasticLayer):
    """
    Negative binomial layer with scaled Poisson mean.
    """

    def __init__(self, in_features, out_features, disp="feature-sample", n_batches=None):
        super().__init__(in_features, out_features)

        self.disp = disp
        self.p_mean = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.Softmax(dim=-1))
        if disp == 'feature-sample':
            self.p_total_count = nn.Linear(self.in_features, self.out_features)
        elif disp == 'feature':
            self.p_total_count = torch.nn.Parameter(torch.ones(self.out_features))
        elif disp == 'feature-batch':
            if n_batches is None:
                raise ValueError('Need to specify n_batches when using feature-batch dispersion!')
            self.p_total_count = torch.nn.Parameter(torch.ones((n_batches, self.out_features)))
        else:
            raise ValueError('Unknown dispersion setting: ' + str(disp))


    def forward(self, x, l, y=None):
        p_mean = self.p_mean(x)
        if self.disp == 'feature-sample':
            total_count = self.p_total_count(x).exp()
        if self.disp == 'feature':
            total_count = self.p_total_count.expand(p_mean.shape)
        if self.disp == 'feature-batch':
            total_count = self.p_total_count[y, :]
        p_mean = p_mean * l
        probs = p_mean / (p_mean + total_count)

        assert (p_mean > 1e-5).any() and (p_mean < 1e+3).any()
        assert (total_count > 1e-5).any() and (total_count < 1e+3).any()

        assert not torch.isnan(p_mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class ScaledPoissonSL(StochasticLayer):
    """
    Poisson model with scaled mean.
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

        self.p_mean = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.Softmax(dim=-1))


    def forward(self, x, l):
        p_mean = self.p_mean(x)
        p_mean = p_mean * l

        assert (p_mean > 1e-5).any() and (p_mean < 1e+3).any()

        assert not torch.isnan(p_mean).any()

        return tdist.Independent(tdist.Poisson(rate=p_mean), 1)

