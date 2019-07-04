import collections

import numpy as np

from . import base


__all__ = ['FTRLProximal']


class FTRLProximal(base.Optimizer):
    """FTRL-Proximal optimizer.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> optimizer = optim.FTRLProximal()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.963585

    References:
        1. `Ad Click Prediction: a View from the Trenches - Research - Google <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf>`_
        2. `tf.train.FtrlOptimizer <https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer>`_

    """

    def __init__(self, alpha=.05, beta=1., l1=0., l2=1.):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = collections.defaultdict(float)
        self.n = collections.defaultdict(float)

    def update_after_pred(self, w, g):

        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2
        z = self.z
        n = self.n

        for i in g:
            if abs(z[i]) > l1:
                w[i] = -((beta + n[i] ** 0.5) / alpha + l2) ** -1 * (z[i] - np.sign(z[i]) * l1)

        for i, gi in g.items():
            s = ((self.n[i] + gi ** 2) ** 0.5 - self.n[i] ** 0.5) / self.alpha
            self.z[i] += gi - s * w.get(i, 0)
            self.n[i] += gi ** 2

        return w
