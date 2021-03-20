import collections

from . import base

__all__ = ["NesterovMomentum"]


class NesterovMomentum(base.Optimizer):
    """Nesterov Momentum optimizer.

    Parameters
    ----------
    lr
    rho

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.NesterovMomentum()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.842932

    """

    def __init__(self, lr=0.1, rho=0.9):
        super().__init__(lr)
        self.rho = rho
        self.s = collections.defaultdict(float)

    def look_ahead(self, w):

        for i in w:
            w[i] -= self.rho * self.s[i]

        return w

    def _step(self, w, g):

        # Move w back to it's initial position
        for i in w:
            w[i] += self.rho * self.s[i]

        for i, gi in g.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]

        return w
