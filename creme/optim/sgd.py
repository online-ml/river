from creme import utils

from . import base


__all__ = ['SGD']


class SGD(base.Optimizer):
    """Plain stochastic gradient descent.

    Parameters:
        lr

    Example:

        >>> from creme import datasets
        >>> from creme import evaluate
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import optim
        >>> from creme import preprocessing

        >>> dataset = datasets.Phishing()
        >>> optimizer = optim.SGD(0.1)
        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     linear_model.LogisticRegression(optimizer)
        ... )
        >>> metric = metrics.F1()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        F1: 0.878521

    References:
        1. [Robbins, H. and Monro, S., 1951. A stochastic approximation method. The annals of mathematical statistics, pp.400-407](https://pdfs.semanticscholar.org/34dd/d8865569c2c32dec9bf7ffc817ff42faaa01.pdf)

    """

    def __init__(self, lr=.01):
        super().__init__(lr)

    def _update_after_pred(self, w, g, h):

        if isinstance(w, utils.VectorDict) and isinstance(g, utils.VectorDict):
            w -= self.learning_rate * g
        else:
            for i, gi in g.items():
                w[i] -= self.learning_rate * gi

        return w
