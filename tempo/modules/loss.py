# -*- coding: utf-8 -*-
"Loss functions."

import torch

from torch.nn.modules.loss import _Loss


class ELBO(_Loss):

    def __init__(self, model, beta=1, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.model = model
        self.beta = beta

        # attributes for tracking
        self.kl_div = 0


    def forward(self, output, target):
        log_lik, self.kl_div = output

        assert not torch.isnan(self.kl_div).any(), 'KL divergence is nan!'
        assert not torch.isnan(log_lik).any(), 'LL is nan!'

        if self.beta > 0:
            ret = - (log_lik - self.beta * self.kl_div)
        else:
            ret = - log_lik
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret

