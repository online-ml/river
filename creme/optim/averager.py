import collections

from . import base


__all__ = ['Averager']


class Averager(base.Optimizer):
    """Averaged stochastic gradient descent.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme.optim import Nadam
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> nadam = Nadam()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(nadam)
            ... )
            >>> metric = metrics.F1()
            >>> model_selection.progressive_val_score(X_y, model, metric)
            F1: 0.958217

            >>> from creme.optim import Averager
            >>> avg_nadam = Averager(Nadam())
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(avg_nadam)
            ... )
            >>> metric = metrics.F1()
            >>> model_selection.progressive_val_score(X_y, model, metric)
            F1: 0.960894

    References:

    """

    def __init__(self, base_optim):
        self.base_optim = base_optim
        self.w_avg = collections.defaultdict(float)
        self.n_iterations = 0

    def update_before_pred(self, w: dict) -> dict:
        return self.base_optim.update_before_pred(w)

    def _update_after_pred(self, w: dict, g: dict) -> dict:

        # Update the weights
        w = self.base_optim._update_after_pred(w, g)

        # Update the average weights
        for i, wi in w.items():
            self.w_avg[i] = (self.w_avg[i] * self.n_iterations +
                             wi) / (self.n_iterations + 1)

        # # Update the iteration counter
        # self.n_iterations += 1

        return self.w_avg

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'
