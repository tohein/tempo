# -*- coding: utf-8 -*-
"Single-cell variational inference."

import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Normal, LogNormal, Independent
from torch.distributions import kl_divergence

from tempo.utils import one_hot
from tempo.modules.composite import LinearLayers, create_stochastic_net
from tempo.modules.stochastic import NormalSL, LogNormalSL
from tempo.modules.stochastic import  ScaledBernoulliSL
from tempo.modules.stochastic import SingleScaleNormalIntervalsSL, FixedScaleNormalIntervalsSL
from tempo.distributions.multivariate import Multivariate


class Tempo(nn.Module):
    """VAE with time decoder."""

    def __init__(
            self,
            dim_x,
            dim_z,
            n_windows,
            l_locs,
            l_scales,
            d_hidden_encoder_z,
            d_hidden_encoder_l,
            d_hidden_decoder_t,
            d_hidden_decoder_x=None,
            likelihood_layer=ScaledBernoulliSL,
            learn_time_variance=False,
            mc_samples=5):
        super().__init__()

        # dimension of latent and observed space
        self.dim_z = dim_z
        self.dim_x = dim_x

        # number of batches and windows
        self.n_batches = l_locs.shape[0]
        self.n_windows = n_windows

        # init windows (-inf, -1, ..., 1, inf), with equal spacing between -1 and 1
        delta = 2 / (self.n_windows - 2)
        self.register_parameter('interval_1', torch.nn.Parameter(torch.Tensor([-1])))
        self.register_parameter('deltas', torch.nn.Parameter(delta * torch.ones(self.n_windows - 2)))

        self.register_buffer('interval_start', torch.Tensor([-np.inf]))
        self.register_buffer('interval_end', torch.Tensor([np.inf]))

        self.encoder_z = create_stochastic_net(dim_x + self.n_batches, d_hidden_encoder_z, dim_z, NormalSL)
        self.encoder_l = create_stochastic_net(dim_x + self.n_batches, d_hidden_encoder_l, 1, LogNormalSL)

        t_stochastic_layer = FixedScaleNormalIntervalsSL
        if learn_time_variance:
            t_stochastic_layer = SingleScaleNormalIntervalsSL

        if d_hidden_decoder_t is None:
            self.decoder_t = torch.nn.modules.Identity()
            self.decoder_t_sl = t_stochastic_layer(self.dim_z)
        else:
            self.decoder_t = LinearLayers([self.dim_z, *d_hidden_decoder_t])
            self.decoder_t_sl = t_stochastic_layer(d_hidden_decoder_t[-1])

        if d_hidden_decoder_x is None:
            self.decoder_net = torch.nn.modules.Identity()
            self.decoder_sl = likelihood_layer(dim_z + self.n_batches, dim_x)
        else:
            self.decoder_net = LinearLayers([dim_z + self.n_batches, *d_hidden_decoder_x])
            self.decoder_sl = likelihood_layer(d_hidden_decoder_x[-1], dim_x)

        self.register_buffer('z_loc', torch.zeros(1, dim_z))
        self.register_buffer('z_scale', torch.ones(1, dim_z))
        self.register_buffer('l_locs', torch.Tensor(l_locs))
        self.register_buffer('l_scales', torch.Tensor(l_scales))

        # monte-carlo sampels for ELBO evaluation
        self.mc_samples = mc_samples


    def forward(self, X):
        x, b, w = self.split_observed(X)

        # b, w will be used as indices, squeeze
        b, w = b.squeeze(), w.squeeze()
        batch_size = X.shape[0]

        l_post = self.posterior_l(x, b)
        z_post = self.posterior_z(x, b)

        # these terms do not require mc sampling
        # ======================================

        # size factor kl divergence
        kl_l = kl_divergence(l_post, self.prior_l(b))
        kl_z = kl_divergence(z_post, self.prior_z(batch_size))

        assert not torch.isnan(kl_l).any(), 'KL divergence is nan!'
        assert not torch.isnan(kl_z).any(), 'KL divergence is nan!'

        # all below depends on samples from z
        # ===================================

        b = b.repeat([self.mc_samples])
        w = w.repeat([self.mc_samples])
        x = x.repeat([self.mc_samples, 1])

        windows = self.create_windows(w)

        # sample and treat as batch dimension
        z = z_post.rsample(sample_shape=[self.mc_samples]).flatten(0, 1)
        l = l_post.rsample(sample_shape=[self.mc_samples]).flatten(0, 1)

        # compute likelihood and average over mc samples
        x_dist = self.likelihood(z, l, b)
        x_log_lik = x_dist.log_prob(torch.cat([x, windows], dim=-1))
        x_log_lik = x_log_lik.reshape([self.mc_samples, batch_size]).mean(0)

        return x_log_lik, kl_l + kl_z


    def split_observed(self, X):
        x, b, w = X.split([self.dim_x, 1, 1], -1)
        return x, b.long(), w.long()


    def create_windows(self, w):
        windows = torch.cat([
            self.interval_start,
            self.interval_1,
            self.interval_1 + self.deltas.cumsum(0),
            self.interval_end])
        return torch.stack([windows[w], windows[w+1]], dim=1)


    # posteriors
    # ==========

    def posterior_z(self, x, b):
        z_post = self.encoder_z(torch.cat([x, one_hot(b, self.n_batches)], dim=-1))
        return z_post


    def posterior_l(self, x, b):
        l_post = self.encoder_l(torch.cat([x, one_hot(b, self.n_batches)], dim=-1))
        return l_post


    # priors
    # ======

    def prior_l(self, b):
        l_loc, l_scale = self.l_locs[b, :], self.l_scales[b, :]
        l_prior = Independent(LogNormal(l_loc, l_scale), 1)
        return l_prior


    def prior_z(self, batch_size):
        z_prior = Independent(Normal(self.z_loc, self.z_scale), 1)
        z_prior = z_prior.expand(torch.Size([batch_size]))
        return z_prior


    # likelihood
    # ==========

    def likelihood(self, z, l, b):
        # decode
        b_one_hot = one_hot(b, self.n_batches)
        zb_transformed_x = self.decoder_net(torch.cat([z, b_one_hot], dim=-1))
        zb_transformed_t = self.decoder_t(z)

        return Multivariate([
            self.decoder_sl(zb_transformed_x, l, b),
            self.decoder_t_sl(zb_transformed_t)])

    # other
    # =====

    def compute_activities(self, z, b):
        # compute activities for scaled stochastic layers
        b_one_hot = one_hot(b, self.n_batches)
        zb_transformed_x = self.decoder_net(torch.cat([z, b_one_hot], dim=-1))
        activities = self.decoder_sl.p_mean(zb_transformed_x)
        return activities

