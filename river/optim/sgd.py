from __future__ import annotations

from river import optim

__all__ = ["SGD"]


class SGD(optim.base.Optimizer):
    """Plain stochastic gradient descent.

    Parameters
    ----------
    lr

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.SGD(0.1)
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 87.85%

    References
    ----------
    [^1]: [Robbins, H. and Monro, S., 1951. A stochastic approximation method. The annals of mathematical statistics, pp.400-407](https://pdfs.semanticscholar.org/34dd/d8865569c2c32dec9bf7ffc817ff42faaa01.pdf)

    """

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def _step_with_dict(self, w, g):
        for i, gi in g.items():
            w[i] -= self.learning_rate * gi
        return w

    def _step_with_vector(self, w, g):
        w -= self.learning_rate * g
        return w
