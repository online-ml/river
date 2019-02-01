import collections

from . import base


__all__ = ['AdaGrad']


class AdaGrad(base.Optimizer):
    """AdaGrad optimizer.

    Example:

    ::

        >>> import creme.compose
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.optim
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = creme.optim.AdaGrad()
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.992977...

    """

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.g2 = collections.defaultdict(lambda: 0.)

    def update_weights_with_gradient(self, w, g):

        for i, gi in g.items():
            self.g2[i] += gi ** 2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w
