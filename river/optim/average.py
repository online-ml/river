import collections

from . import base


class Averager(base.Optimizer):
    """Averaged stochastic gradient descent.

    This is a wrapper that can be applied to any stochastic gradient descent optimiser. Note that
    this implementation differs than what may be found elsewhere. Essentially, the average of the
    weights is usually only used at the end of the optimisation, once all the data has been seen.
    However, in this implementation the optimiser returns the current averaged weights.

    Parameters
    ----------
    optimizer
        An optimizer for which the produced weights will be averaged.
    start
        Indicates the number of iterations to wait before starting the average. Essentially,
        nothing happens differently before the number of iterations reaches this value.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.Averager(optim.SGD(0.01), 100)
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.878924

    References
    ----------
    [^1]: [Bottou, L., 2010. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010 (pp. 177-186). Physica-Verlag HD.](https://leon.bottou.org/publications/pdf/compstat-2010.pdf)
    [^2]: [Stochastic Algorithms for One-Pass Learning slides by Léon Bottou](https://leon.bottou.org/slides/onepass/onepass.pdf)
    [^3]: [Xu, W., 2011. Towards optimal one pass large scale learning with averaged stochastic gradient descent. arXiv preprint arXiv:1107.2490.](https://arxiv.org/pdf/1107.2490.pdf)

    """

    def __init__(self, optimizer: base.Optimizer, start: int = 0):
        self.optimizer = optimizer
        self.start = start
        self.avg_w = collections.defaultdict(float)
        self.n_iterations = 0

    def look_ahead(self, w):
        return self.optimizer.look_ahead(w)

    def _step(self, w, g):

        w = self.optimizer.step(w, g)

        # No averaging occurs during the first start iterations
        if self.n_iterations < self.start:
            return w

        for i, wi in w.items():
            self.avg_w[i] += (wi - self.avg_w[i]) / (self.n_iterations - self.start + 1)

        return self.avg_w
