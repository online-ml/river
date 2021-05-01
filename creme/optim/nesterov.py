import collections

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
        self.s = collections.defaultdict(float)

    def update_before_pred(self, w):

        for i in w:
            w[i] -= self.rho * self.s[i]

        return w

    def _update_after_pred(self, w, g, h):

        # Move w back to it's initial position
        for i in w:
            w[i] += self.rho * self.s[i]

        for i, gi in g.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]

        return w
