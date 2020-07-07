import torch
import torch.distributions as tdist


def one_hot(index, n_cat):
    if index.ndim == 1:
        index = index.reshape(-1, 1)
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def run_3d_helper(fun, args, param_list):
    """Runs stochastic net on 3 dimensional inputs.

    2-dimensional inputs are repeated. For 3 dimensional args,
    the first dimension is integrated into the second dimension before
    concatenating the arguments and passing them through the network.
    Finally the output is reshaped to reflect the additional dimension.

    Only works for TorchDistributionSL output layers.

    No argument checks!

    Args:
        snet: Stochastic net.
        args (list): Tensor inputs.
        param_list (list): List of parameters of the distribution.
    """
    # function to transform 2d and 3d tensors to 2d
    def _transform_2d(x, size):
        if x.ndim == 2:
            return x.repeat(size, 1)
        else:
            return x.reshape(size * x.shape[1], x.shape[2])

    # determine size of dimension to be integrated
    size = None
    for arg in args:
        if arg.ndim != 3:
            continue
        if size is None:
            size = arg.shape[0]
        elif arg.shape[0] != size:
            raise ValueError('Inconsistent tensor shapes.')
    # run net on concatenated reshaped inputs
    out = fun(*[_transform_2d(x, size) for x in args])
    batch_size = out.batch_shape[0] // size
    # reshape distribution parameters
    kwargs = dict()
    for p in param_list:
        kwargs[p] = getattr(out.base_dist, p).reshape(size, batch_size, out.event_shape[0])
    # create new distribution object with appropriate dimensions
    return tdist.Independent(type(out.base_dist)(**kwargs), 1)
