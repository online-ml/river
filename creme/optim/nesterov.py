import collections

from . import base


__all__ = ['NesterovMomentum']


class NesterovMomentum(base.Optimizer):
    """Nesterov Momentum optimizer.

    Example:

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
        >>> optimiser = creme.optim.NesterovMomentum()
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.975860...

    """

    def __init__(self, lr=0.1, rho=0.9):
        super().__init__(lr)
        self.rho = rho
        self.s = collections.defaultdict(lambda: 0.)

    def update_weights(self, x, y, w, f_pred, f_grad):

        # Move the weights to the future position
        for i in w:
            w[i] -= self.rho * self.s[i]

        # Compute the gradient
        y_pred = f_pred(x, w)
        gradient = f_grad(y, y_pred, x, w)

        # Update the step and the weights
        for i, gi in gradient.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]

        return w, y_pred
