import collections

from . import base

__all__ = ["AdaGrad"]


class AdaGrad(base.Optimizer):
    """AdaGrad optimizer.

    Parameters
    ----------
    lr
    eps

    Attributes
    ----------
    g2 : collections.defaultdict

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.AdaGrad()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.880143

    References
    ----------
    [^1]: [Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(Jul), pp.2121-2159.](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

    """

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.g2 = collections.defaultdict(float)

    def _step(self, w, g):

        for i, gi in g.items():
            self.g2[i] += gi ** 2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w
