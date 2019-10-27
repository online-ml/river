import collections
import copy
import statistics

from sklearn import utils

from .. import base


__all__ = ['BaggingClassifier', 'BaggingRegressor']


class BaseBagging(base.Wrapper, base.Ensemble):

    def __init__(self, model, n_models=10, random_state=None):
        super().__init__(copy.deepcopy(model) for i in range(n_models))
        self.model = model
        self.rng = utils.check_random_state(random_state)

    @property
    def _model(self):
        return self.model

    def fit_one(self, x, y):

        for model in self:
            for _ in range(self.rng.poisson(1)):
                model.fit_one(x, y)

        return self


class BaggingClassifier(BaseBagging, base.Classifier):
    """Bagging for classification.

    For each incoming observation, each model's ``fit_one`` method is called ``k`` times where
    ``k`` is sampled from a Poisson distribution of parameter 1. ``k`` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    ``scipy.stats.poisson(1).pmf(k)`` to obtain more detailed values.

    Parameters:
        model (BinaryClassifier or MultiClassifier): The classifier to bag.
        n_models (int): The number of models in the ensemble.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Example:

        In the following example three logistic regressions are bagged together. The performance is
        slightly better than when using a single logistic regression.

        ::

            >>> from creme import compose
            >>> from creme import ensemble
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
            >>> model = ensemble.BaggingClassifier(
            ...     model=(
            ...         preprocessing.StandardScaler() |
            ...         linear_model.LogisticRegression()
            ...     ),
            ...     n_models=3,
            ...     random_state=42
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.966667

            >>> print(model)
            BaggingClassifier(StandardScaler | LogisticRegression)

    References:
        1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        return {label: proba / total for label, proba in y_pred.items()}


class BaggingRegressor(BaseBagging, base.Regressor):
    """Bagging for regression.

    For each incoming observation, each model's ``fit_one`` method is called ``k`` times where
    ``k`` is sampled from a Poisson distribution of parameter 1. ``k`` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    ``scipy.stats.poisson(1).pmf(k)`` for more detailed values.

    Parameters:
        model (Regressor): The regressor to bag.
        n_models (int): The number of models in the ensemble.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Example:

        In the following example three logistic regressions are bagged together. The performance is
        slightly better than when using a single logistic regression.

        ::

            >>> from creme import compose
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = preprocessing.StandardScaler()
            >>> model |= ensemble.BaggingRegressor(
            ...     model=linear_model.LinearRegression(intercept_lr=0.1),
            ...     n_models=3,
            ...     random_state=42
            ... )
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 4.26101

    References:
        1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def predict_one(self, x):
        """Averages the predictions of each regressor."""
        return statistics.mean((regressor.predict_one(x) for regressor in self))
