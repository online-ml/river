from __future__ import annotations

import collections

import numpy as np

from river import optim

__all__ = ["FTRLProximal"]


class FTRLProximal(optim.base.Optimizer):
    """FTRL-Proximal optimizer.

    Parameters
    ----------
    alpha
    beta
    l1
    l2

    Attributes
    ----------
    z : collections.defaultdict
    n : collections.defaultdict

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.FTRLProximal()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 87.56%

    References
    ----------
    [^1]: [McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D. and Chikkerur, S., 2013, August. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222-1230)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
    [^2]: [Tensorflow's `FtrlOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer)

    """

    def __init__(self, alpha=0.05, beta=1.0, l1=0.0, l2=1.0):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = collections.defaultdict(float)
        self.n = collections.defaultdict(float)
        self.n_iterations = 0

    def _step_with_dict(self, w, g):
        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2
        z = self.z
        n = self.n

        for i in g:
            if abs(z[i]) > l1:
                w[i] = -(((beta + n[i] ** 0.5) / alpha + l2) ** -1) * (z[i] - np.sign(z[i]) * l1)

        for i, gi in g.items():
            s = ((self.n[i] + gi**2) ** 0.5 - self.n[i] ** 0.5) / self.alpha
            self.z[i] += gi - s * w.get(i, 0)
            self.n[i] += gi**2

        return w
