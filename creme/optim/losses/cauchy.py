from . import base


class CauchyLoss(base.RegressionLoss):
    """Cauchy loss function.

    References:
        1. `Effect of MAE <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163>`_
        2. `Paris Madness <https://www.kaggle.com/raddar/paris-madness>`_

    """

    def __init__(self, C=80):
        self.C = C

    def __call__(self, y_true, y_pred):
        return abs(y_pred - y_true)

    def gradient(self, y_true, y_pred):
        diff = y_pred - y_true
        return diff / ((diff / self.C) ** 2 + 1)
