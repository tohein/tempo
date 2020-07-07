import torch
from torch.distributions import Distribution
from torch.distributions.kl import register_kl, kl_divergence


class Multivariate(Distribution):
    """
    Bundles multiple torch.Distribution objects into a single distribution.

    This class is used to combine multiple independent distributions and
    generalizes basic functions such as their Kullback-Leibler divergence,
    rsample and log_prob.

    All base distributions are expected to have the same batch dimensions and
    zero- or one-dimensional event shapes.
    """
    def __init__(self, base_distributions, validate_args=None):
        self.event_sizes = []
        event_shape = 0
        batch_shape = base_distributions[0].batch_shape
        for dist in base_distributions:
            if dist.batch_shape != batch_shape:
                raise ValueError("Expected all base distributions to have the same batch dimensions.")
            if len(dist.event_shape) == 0:
                self.event_sizes.append(1)
            elif len(dist.event_shape) == 1:
                self.event_sizes.append(dist.event_shape[0])
            else:
                raise ValueError("Expected all base distributions to have zero- or one-dimensional event shapes.")

        event_shape = torch.Size([sum(self.event_sizes)])
        self.base_distributions = base_distributions
        super().__init__(batch_shape, event_shape, validate_args=validate_args)


    def _combine_res(self, res):
        if self.batch_shape == torch.Size([]):
            return res[0].new_tensor(torch.Tensor(res))
        if len(res[0].shape) == 1:
            return torch.stack(res, 1)
        else:
            return torch.cat(res, -1)


    def _call_base_function(self, fun, *args, indiv_args=None, **kwargs):
        res = []
        if indiv_args is not None:
            # TODO implement argument checks ...
            indiv_args = indiv_args.split(self.event_sizes, -1)
        for i, dist in enumerate(self.base_distributions):
            if indiv_args is not None:
                indiv_arg = indiv_args[i]
                res.append(getattr(dist, fun)(indiv_arg, *args, **kwargs))
            else:
                res.append(getattr(dist, fun)(*args, **kwargs))
        return self._combine_res(res)


    def _get_base_attribute(self, attr):
        res = []
        for dist in self.base_distributions:
            res.append(getattr(dist, attr))
        self._combine_res(res)


    @property
    def has_rsample(self):
        return all(self._get_base_attribute("has_rsample"))


    @property
    def mean(self):
        return self._get_base_attribute("mean")


    @property
    def variance(self):
        return self._get_base_attribute("variance")


    def sample(self, sample_shape=torch.Size()):
        return self._call_base_function("sample", sample_shape)


    def rsample(self, sample_shape=torch.Size()):
        return self._call_base_function("rsample", sample_shape)


    def log_prob(self, values):
        res = self._call_base_function("log_prob", indiv_args=values)
        return res.sum(1)


    def cdf(self, values):
        res = self._call_base_function("cdf", indiv_args=values)
        return res.prod(0)


    def entropy(self):
        res = self._call_base_function("entropy")
        return res.sum(0)


@register_kl(Multivariate, Multivariate)
def _kl_independent_independent(p, q):
    kl_div = 0
    for i, dist in enumerate(p.base_distributions):
        kl_div += kl_divergence(dist, q.base_distributions[i])
    return kl_div
