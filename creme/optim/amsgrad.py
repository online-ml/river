import collections

from . import base


__all__ = ['AMSGrad']


class AMSGrad(base.Optimizer):
    """AMSGrad optimizer.

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
            >>> optimizer = optim.AMSGrad()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            F1: 0.957983

    References:
        1. `ON THE CONVERGENCE OF ADAM AND BEYOND <https://arxiv.org/pdf/1904.09237.pdf>`_

    """

    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8, correct_bias=True):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.correct_bias = correct_bias
        self.m = collections.defaultdict(float)
        self.v = collections.defaultdict(float)
        self.v_hat = collections.defaultdict(float)

    def _update_after_pred(self, w, g):

        if self.correct_bias:
            # Correct bias for `v`
            learning_rate = self.learning_rate * (1 - self.beta_2 ** (self.n_iterations + 1))**0.5
            # Correct bias for `m`
            learning_rate /= 1 - self.beta_1 ** (self.n_iterations + 1)
        else:
            learning_rate = self.learning_rate

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * gi ** 2
            self.v_hat[i] = max(self.v_hat[i], self.v[i])

            w[i] -= learning_rate * self.m[i] / (self.v_hat[i] ** 0.5 + self.eps)

        return w
