import collections

import numpy as np

from . import base


__all__ = ['FTRLProximal']


class FTRLProximal(base.Optimizer):
    """FTRL-Proximal optimizer.

    Example:

    ::

        >>> from creme import compose
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import optim
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = optim.FTRLProximal()
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.964639

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

    def update_weights(self, x, y, w, loss, f_pred, f_grad):

        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2
        z = self.z
        n = self.n
        w = {}

        for i, xi in x.items():
            if abs(z[i]) > l1:
                w[i] = -((beta + n[i] ** 0.5) / alpha + l2) ** -1 * (z[i] - np.sign(z[i]) * l1)

        y_pred = f_pred(x, w)
        gradient = f_grad(y_true=y, y_pred=y_pred, loss=loss, x=x, w=w)

        for i, gi in gradient.items():
            s = ((n[i] + gi ** 2) ** 0.5 - n[i] ** 0.5) / alpha
            z[i] += gi - s * w.get(i, 0)
            n[i] += gi ** 2

        return w, y_pred
