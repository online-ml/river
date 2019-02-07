import collections

from . import base


__all__ = ['AdaDelta']


class AdaDelta(base.Optimizer):
    """AdaDelta optimizer.

    Example:

    ::

        >>> import creme.compose
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.optim
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = creme.optim.AdaDelta()
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.977386...

    References:

    - `AdaDelta: an adaptive learning rate method <https://arxiv.org/pdf/1212.5701.pdf>`_

    """

    def __init__(self, rho=0.95, eps=1e-8):
        super().__init__(lr=None)
        self.rho = rho
        self.eps = eps
        self.g2 = collections.defaultdict(lambda: 0.)
        self.s2 = collections.defaultdict(lambda: 0.)

    def _rms(self, x):
        return (x + self.eps) ** 0.5

    def update_weights_with_gradient(self, w, g):

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
