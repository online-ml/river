import collections

from . import base


__all__ = ['NesterovMomentum']


class NesterovMomentum(base.Optimizer):
    """Nesterov Momentum optimizer.

    Example:

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
        >>> optimiser = optim.NesterovMomentum()
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', linear_model.LogisticRegression(optimiser))
        ... ])
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.950774

    """

    def __init__(self, lr=0.1, rho=0.9):
        super().__init__(lr)
        self.rho = rho
        self.s = collections.defaultdict(float)

    def update_weights(self, x, y, w, loss, f_pred, f_grad):

        # Move the weights to the future position
        for i in w:
            w[i] -= self.rho * self.s[i]

        # Compute the gradient
        y_pred = f_pred(x, w)
        gradient = f_grad(y, y_pred, loss, x, w)

        # Update the step and the weights
        for i, gi in gradient.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]

        return w, y_pred
