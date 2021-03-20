import collections
import typing

from . import base, schedulers

__all__ = ["AMSGrad"]


class AMSGrad(base.Optimizer):
    """AMSGrad optimizer.

    Parameters
    ----------
    lr
        The learning rate.
    beta_1
    beta_2
    eps
    correct_bias

    Attributes
    ----------
    m : collections.defaultdict
    v : collections.defaultdict
    v_hat : collections.defaultdict

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.AMSGrad()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.865724

    References
    ----------
    [^1]: [Reddi, S.J., Kale, S. and Kumar, S., 2019. On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237](https://arxiv.org/pdf/1904.09237.pdf)

    """

    def __init__(
        self,
        lr: typing.Union[float, schedulers.Scheduler] = 0.1,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        correct_bias=True,
    ):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.correct_bias = correct_bias
        self.m = collections.defaultdict(float)
        self.v = collections.defaultdict(float)
        self.v_hat = collections.defaultdict(float)

    def _step(self, w, g):

        lr = self.learning_rate

        if self.correct_bias:
            # Correct bias for `v`
            lr *= (1 - self.beta_2 ** (self.n_iterations + 1)) ** 0.5
            # Correct bias for `m`
            lr /= 1 - self.beta_1 ** (self.n_iterations + 1)

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * gi ** 2
            self.v_hat[i] = max(self.v_hat[i], self.v[i])

            w[i] -= lr * self.m[i] / (self.v_hat[i] ** 0.5 + self.eps)

        return w
