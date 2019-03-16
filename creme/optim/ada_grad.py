import collections

from . import base


__all__ = ['AdaGrad']


class AdaGrad(base.Optimizer):
    """AdaGrad optimizer.

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
        >>> optimiser = optim.AdaGrad()
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.97191

    """

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.g2 = collections.defaultdict(float)

    def update_weights_with_gradient(self, w, g):

        for i, gi in g.items():
            self.g2[i] += gi ** 2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w
