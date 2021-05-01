import collections

from . import base


__all__ = ['RMSProp']


class RMSProp(base.Optimizer):
    """RMSProp optimizer.

    Parameters:
        lr
        rho
        eps

    Example:

        >>> from creme import datasets
        >>> from creme import evaluate
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import optim
        >>> from creme import preprocessing

        >>> dataset = datasets.Phishing()
        >>> optimizer = optim.RMSProp()
        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     linear_model.LogisticRegression(optimizer)
        ... )
        >>> metric = metrics.F1()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        F1: 0.872378

    References:
        1. [Divide the gradient by a running average of itsrecent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    """

    def __init__(self, lr=.1, rho=.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.g2 = collections.defaultdict(float)

    def _update_after_pred(self, w, g, h):

        for i, gi in g.items():
            self.g2[i] = self.rho * self.g2[i] + (1 - self.rho) * gi ** 2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w
