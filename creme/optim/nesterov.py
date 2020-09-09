from creme import optim
from creme import utils

from . import base


__all__ = ['NesterovMomentum']


class NesterovMomentum(base.Optimizer):
    """Nesterov Momentum optimizer.

    Parameters:
        lr
        rho

    Example:

        >>> from creme import datasets
        >>> from creme import evaluate
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import optim
        >>> from creme import preprocessing

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

    def __init__(self, lr=.1, rho=.9):
        super().__init__(lr)
        self.rho = rho
        self.s = utils.VectorDict(None, optim.initializers.Zeros())

    def update_before_pred(self, w):
        w -= self.rho * self.s
        return w

    def _update_after_pred(self, w, g):

        # Move w back to it's initial position
        w += self.rho * self.s

        if (isinstance(w, utils.VectorDict) and isinstance(g, utils.VectorDict) and
                g.keys() == w.keys()):
            self.s = self.rho * self.s + self.learning_rate * g
            w -= self.s
        else:
            for i, gi in g.items():
                self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
                w[i] -= self.s[i]

        return w
