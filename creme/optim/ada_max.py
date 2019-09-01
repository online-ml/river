import collections

from . import base


__all__ = ['AdaMax']


class AdaMax(base.Optimizer):
    """AdaMax optimizer.

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
            >>> optimizer = optim.AdaMax()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.97479

    References:
        1. `Adam: A method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_
        2. `An overview of gradient descent optimization algorithms <http://ruder.io/optimizing-gradient-descent/index.html#adamax>`_

    """

    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = collections.defaultdict(float)
        self.u = collections.defaultdict(float)

    def _update_after_pred(self, w, g):

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            self.u[i] = max(self.beta_2 * self.u[i], abs(gi))
            m = self.m[i] / (1 - self.beta_1 ** (self.n_iterations + 1))
            w[i] -= self.learning_rate * m / (self.u[i] + self.eps)

        return w
