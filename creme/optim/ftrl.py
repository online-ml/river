import collections

import numpy as np

from . import base


__all__ = ['FTRLProximal']


class FTRLProximal(base.Optimizer):
    """
    Example
    -------

        #!python
        >>> import creme
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = creme.optim.FTRLProximal()
        >>> model = creme.pipeline.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.989859...

    """

    def __init__(self, alpha=.05, beta=1., l1=0., l2=1.):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = collections.defaultdict(lambda: 0.)
        self.n = collections.defaultdict(lambda: 0.)

    def update_weights(self, x, y, w, f_pred, f_grad):

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
        gradient = f_grad(y, y_pred, x, w)

        for i, gi in gradient.items():
            s = ((n[i] + gi ** 2) ** 0.5 - n[i] ** 0.5) / alpha
            z[i] += gi - s * w.get(i, 0)
            n[i] += gi ** 2

        return w, y_pred
