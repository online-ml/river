import collections

from . import base


__all__ = ["AdaDelta"]


class AdaDelta(base.Optimizer):
    """AdaDelta optimizer.

    Parameters
    ----------
    rho
    eps

    Attributes
    ----------
    g2 : collections.defaultdict
    s2 : collections.defaultdict

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.AdaDelta()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.805611

    References
    ----------
    [^1]: [Zeiler, M.D., 2012. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.](https://arxiv.org/pdf/1212.5701.pdf)

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
            step = -self._rms(self.s2[i]) / self._rms(self.g2[i]) * gi

            # Accumulate the update
            self.s2[i] = self.rho * self.s2[i] + (1 - self.rho) * step ** 2

            # Apply the update
            w[i] += step

        return w
