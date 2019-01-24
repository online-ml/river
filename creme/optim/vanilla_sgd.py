from . import base


__all__ = ['VanillaSGD']


class VanillaSGD(base.Optimizer):
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
        >>> optimiser = creme.optim.VanillaSGD()
        >>> model = creme.pipeline.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.990625...

    """

    def __init__(self, lr=0.1):
        super().__init__(lr)

    def update_weights_with_gradient(self, w, g):

        for i, gi in g.items():
            w[i] -= self.learning_rate * gi

        return w
