from vowpalwabbit import pyvw

from .. import base


__all__ = [
    'VW2CremeRegressor'
]


class VW2CremeBase(pyvw.vw):

    def _format_features(self, x):
        return ' '.join(f'{k}:{v}' for k, v in x.items())

    def fit_one(self, x, y):
        ex = self.example(f'{y} | {self._format_features(x)}')
        self.learn(ex)
        self.finish_example(ex)
        return self


class VW2CremeRegressor(VW2CremeBase, base.Regressor):
    """Vowpal Wabbit to ``creme`` regressor adapter.

    Example:

        ::

            >>> from creme import compat
            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import preprocessing
            >>> from creme import model_selection

            >>> X_y = datasets.TrumpApproval()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     compat.VW2CremeRegressor(sgd=True, quiet=True)
            ... )

            >>> metric = metrics.MAE()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            MAE: 5.76122

    References:
        1. `Vowpal Wabbit parameters <https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-line-arguments>`_

    """

    def predict_one(self, x):
        ex = self.example(f' | {self._format_features(x)}')
        y_pred = self.predict(ex)
        self.finish_example(ex)
        return y_pred
