import collections

from . import base


__all__ = ['AdaDelta']


class AdaDelta(base.Optimizer):
    """AdaDelta optimizer.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> optimizer = optim.AdaDelta()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.94751

    References:

        1. `AdaDelta: an adaptive learning rate method <https://arxiv.org/pdf/1212.5701.pdf>`_

    """

    def __init__(self, rho=0.95, eps=1e-8):
        super().__init__(lr=None)
        self.rho = rho
        self.eps = eps
        self.g2 = collections.defaultdict(float)
        self.s2 = collections.defaultdict(float)

    def _rms(self, x):
        return (x + self.eps) ** 0.5

    def _update_after_pred(self, w, g):

        for i, gi in g.items():

            # Accumulate the gradient
            self.g2[i] = self.rho * self.g2[i] + (1 - self.rho) * gi ** 2

            # Compute the update
            step = - self._rms(self.s2[i]) / self._rms(self.g2[i]) * gi

            # Accumulate the update
            self.s2[i] = self.rho * self.s2[i] + (1 - self.rho) * step ** 2

            # Apply the update
            w[i] += step

        return w
