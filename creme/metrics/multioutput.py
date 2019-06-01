from . import base


__all__ = ['RegressionMultiOutput']


class RegressionMultiOutput(base.MultiOutputRegressionMetric):

    def __init__(self, metric: 'RegressionMetric'):
        self.metric = metric

    @property
    def __class__(self):
        return self.metric.__class__

    def bigger_is_better(self):
        return self.metric.bigger_is_better

    def update(self, y_true, y_pred):
        for i in y_true:
            self.metric.update(y_true[i], y_pred[i])
        return self

    def get(self):
        return self.metric.get()
