import torch
from torch.distributions import Distribution, Independent


class IntervalDistribution(Distribution):
    """
    Convenience wrapper to apply a distribution on the real numbers
    to intervals on the real numbers. Let X be a real-valued random variable.
    Then the probability for an interval [a, b] is:

        P(X in [a,b]) = P(X <= b) - P(X <= a)

    This distribution is intended to be used as a likelihood function and
    offers no further functionality. In particular, it can be used as a
    likelihood for ordinal data by identifying each ordinal level with an
    interval on the real line.
    """
    def __init__(self, base_dist, validate_args=None):
        shape = base_dist.event_shape
        if shape != torch.Size([1]):
            raise ValueError('Expected one-dimensional event shape.')
        self.base_dist = base_dist
        event_shape = torch.Size([2])
        super().__init__(
            base_dist.batch_shape,
            event_shape,
            validate_args=validate_args)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def log_prob(self, value, jitter=1e-4):
        a = value[:, 0].reshape(-1, 1)
        b = value[:, 1].reshape(-1, 1)

        if isinstance(self.base_dist, Independent):
            cdf = self.base_dist.base_dist.cdf
        else:
            cdf = self.base_dist.cdf

        cdf_a = cdf(a)
        cdf_b = cdf(b)

        assert not torch.isnan(cdf_a).any()
        assert not torch.isnan(cdf_b).any()

        p = torch.sum(cdf_b - cdf_a, dim=-1)
        p = torch.where(p <= 0, torch.ones(1, device=a.device) * jitter, p)

        return torch.log(p)
