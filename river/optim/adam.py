import collections

from . import base

__all__ = ["Adam"]


class Adam(base.Optimizer):
    """Adam optimizer.

    Parameters
    ----------
    lr
    beta_1
    beta_2
    eps

    Attributes
    ----------
    m : collections.defaultdict
    v : collections.defaultdict

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.Adam()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.86496

    References
    ----------
    [^1]: [Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.](https://arxiv.org/pdf/1412.6980.pdf)

    """

    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = collections.defaultdict(float)
        self.v = collections.defaultdict(float)

    def _step(self, w, g):

        # Correct bias for `v`
        lr = self.learning_rate * (1 - self.beta_2 ** (self.n_iterations + 1)) ** 0.5
        # Correct bias for `m`
        lr /= 1 - self.beta_1 ** (self.n_iterations + 1)

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * gi ** 2
            w[i] -= lr * self.m[i] / (self.v[i] ** 0.5 + self.eps)

        return w
