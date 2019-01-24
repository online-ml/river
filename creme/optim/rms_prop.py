import collections

from . import base


__all__ = ['RMSProp']


class RMSProp(base.Optimizer):
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
        >>> optimiser = creme.optim.RMSProp()
        >>> model = creme.pipeline.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.991167...

    """

    def __init__(self, lr=0.1, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.g2 = collections.defaultdict(lambda: 0.)

    def update_weights_with_gradient(self, w, g):

        for i, gi in g.items():
            self.g2[i] = self.rho * self.g2[i] + (1 - self.rho) * gi ** 2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w
