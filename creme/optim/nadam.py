import math
import collections

from . import base


__all__ = ['Nadam']


class Nadam(base.Optimizer):
    """Nadam optimizer.

    Parameters
    ----------
    lr
    beta_1
    beta_2
    eps

    Examples
    --------

    >>> from creme import datasets
    >>> from creme import evaluate
    >>> from creme import linear_model
    >>> from creme import metrics
    >>> from creme import optim
    >>> from creme import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.Nadam()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.865961

    References
    ----------
    [^1]: [Nadam: A combination of adam and nesterov](https://ruder.io/optimizing-gradient-descent/index.html#nadam)

    """

    def __init__(self, lr=.1, beta_1=.9, beta_2=.999, eps=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = collections.defaultdict(float)
        self.v = collections.defaultdict(float)

    def _update_after_pred(self, w, g):

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            m_hat = self.m[i] / \
                (1 - math.pow(self.beta_1, self.n_iterations + 1))

            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * gi ** 2
            v_hat = self.v[i] / \
                (1 - math.pow(self.beta_2, self.n_iterations + 1))

            w[i] -= self.learning_rate * (self.beta_1 * m_hat + (1 - self.beta_1) * gi / (
                1 - math.pow(self.beta_1, self.n_iterations + 1))) / (v_hat ** .5 + self.eps)

        return w
