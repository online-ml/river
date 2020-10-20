import collections
import copy
import statistics

import numpy as np

from river import base
from river import linear_model
from river.drift import ADWIN


__all__ = ['BaggingClassifier', 'BaggingRegressor']


class BaseBagging(base.WrapperMixin, base.EnsembleMixin):

    def __init__(self, model, n_models=10, seed=None):
        super().__init__(copy.deepcopy(model) for _ in range(n_models))
        self.n_models = n_models
        self.model = model
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _wrapped_model(self):
        return self.model

    def learn_one(self, x, y):

        for model in self:
            for _ in range(self._rng.poisson(1)):
                model.learn_one(x, y)

        return self


class BaggingClassifier(BaseBagging, base.Classifier):
    """Online bootstrap aggregation for classification.

    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    `scipy.stats.poisson(1).pmf(k)` to obtain more detailed values.

    Parameters
    ----------
    model
        The classifier to bag.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = ensemble.BaggingClassifier(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.877788

    >>> print(model)
    BaggingClassifier(StandardScaler | LogisticRegression)

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super().__init__(model, n_models, seed)

    @classmethod
    def _default_params(cls):
        return {'model': linear_model.LogisticRegression()}

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class BaggingRegressor(BaseBagging, base.Regressor):
    """Online bootstrap aggregation for regression.

    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    `scipy.stats.poisson(1).pmf(k)` for more detailed values.

    Parameters
    ----------
    model
        The regressor to bag.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = preprocessing.StandardScaler()
    >>> model |= ensemble.BaggingRegressor(
    ...     model=linear_model.LinearRegression(intercept_lr=0.1),
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.641799

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Regressor, n_models=10, seed: int = None):
        super().__init__(model, n_models, seed)

    @classmethod
    def _default_params(cls):
        return {'model': linear_model.LinearRegression()}

    def predict_one(self, x):
        """Averages the predictions of each regressor."""
        return statistics.mean((regressor.predict_one(x) for regressor in self))


class ADWINBaggingClassifier(BaggingClassifier):
    """ADWIN Bagging classifier.

    ADWIN Bagging [^1] is the online bagging method of Oza and Russell [^2]
    with the addition of the `ADWIN` algorithm as a change detector. When a
    change is detected, the worst classifier of the ensemble of classifier
    (based on the error estimation by ADWIN) is removed and a new classifier
    is added to the ensemble.

    Parameters
    ----------
    model
        The classifier to bag.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = ensemble.ADWINBaggingClassifier(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.878788

    References
    ----------
    [^1]: Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Richard Kirkby,
    and Ricard Gavaldà. "New ensemble methods for evolving data streams."
    In 15th ACM SIGKDD International Conference on Knowledge Discovery and
    Data Mining, 2009.

    [^2]: Oza, N., Russell, S. "Online bagging and boosting."
    In: Artificial Intelligence and Statistics 2001, pp. 105–112.
    Morgan Kaufmann, 2001.

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super().__init__(model=model, n_models=n_models, seed=seed)
        self._drift_detectors = [copy.deepcopy(ADWIN()) for _ in range(self.n_models)]

    def learn_one(self, x, y):

        change_detected = False
        for i, model in enumerate(self):
            for _ in range(self._rng.poisson(1)):
                model.learn_one(x, y)

            try:
                y_pred = model.predict_one(x)
                error_estimation = self._drift_detectors[i].estimation
                self._drift_detectors[i].update(int(y_pred == y))
                if self._drift_detectors[i].change_detected:
                    if self._drift_detectors[i].estimation > error_estimation:
                        change_detected = True
            except ValueError:
                change_detected = False

        if change_detected:
            max_error_idx = max(range(len(self._drift_detectors)),
                                key=lambda j: self._drift_detectors[j].estimation)
            self.models[max_error_idx] = copy.deepcopy(self.model)
            self._drift_detectors[max_error_idx] = ADWIN()

        return self
