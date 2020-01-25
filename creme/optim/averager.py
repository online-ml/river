import collections

from . import base
from . import schedulers


__all__ = ['Averager']


class Averager(base.Optimizer):
    """An Stochastic weight averaging wrapper around various sgd optimzers.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme.optim.schedulers import Cyclic
            >>> from creme.optim import Adam, Averager
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )

            >>> adam = Adam(lr=Cyclic(1, 0.01, 100))
            >>> optimizer = Averager(adam)
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            F1: 0.959441

    References:
        1. `Averaging Weights Leads to Wider Optima and Better Generalization
            <https://arxiv.org/abs/1803.05407>`_

    """

    def __init__(self, base_optim):
        self.base_optim = base_optim
        self.n_iterations = 0
        self.c = (
            base_optim.lr.c
            if isinstance(base_optim.lr, schedulers.Cyclic) else
            1
        )
        self.w_ = collections.defaultdict(float)
        self.w_swa = collections.defaultdict(float)

    def update_before_pred(self, w: dict) -> dict:
        return self.base_optim.update_before_pred(w)

    def _update_after_pred(self, w: dict, g: dict) -> dict:

        # Update the weights
        self.w_ = self.base_optim._update_after_pred(self.w_.copy(), g)

        if self.n_iterations % self.c == 0:
            # Calculate number of models
            n_models = self.n_iterations/self.c

            # Update the average weights
            for i, wi in self.w_.items():
                self.w_swa[i] = (self.w_swa[i] * n_models +
                                 wi) / (n_models + 1)

        self.base_optim.n_iterations += 1

        return (
            self.w_swa
            if self.n_iterations >= self.c else
            self.w_
        )
