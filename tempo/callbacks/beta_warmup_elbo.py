from fastai.basic_train import LearnerCallback

class BetaWarmupELBO(LearnerCallback):
    "Reweight KL divergence in ELBO."
    def __init__(self, learn, beta_update=1./400):
        super().__init__(learn)
        self.beta_update = beta_update

    def on_epoch_end(self, **kwargs):
        "Increase KL weight."
        if self.beta_update:
            beta = min(1, self.learn.loss_func.beta + self.beta_update)
            self.learn.loss_func.beta = beta
