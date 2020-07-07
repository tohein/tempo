from fastai.basic_train import LearnerCallback
from fastai.torch_core import add_metrics


class LossAttrMetric(LearnerCallback):
    _order=-20 # Needs to run before the recorder

    def __init__(self, learn, attr):
        super().__init__(learn)
        self.attr = attr


    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['loss[' + self.attr + ']'])


    def on_batch_end(self, last_output, train, **kwargs):
        # track max memory usage during the train phase
        if not train:
            val = getattr(self.learn.loss_func, self.attr, 0).detach().cpu()    

            self.count += val.shape[0]
            self.val += val.sum()


    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0, 0


    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)
