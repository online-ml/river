import collections
import copy
import statistics

import numpy as np

from river import base
from river import linear_model
from river.drift import ADWIN


__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "ADWINBaggingClassifier",
    "LeveragingBaggingClassifier",
]


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
    def _unit_test_params(cls):
        return {"model": linear_model.LogisticRegression()}

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
    def _unit_test_params(cls):
        return {"model": linear_model.LinearRegression()}

    def predict_one(self, x):
        """Averages the predictions of each regressor."""
        return statistics.mean((regressor.predict_one(x) for regressor in self))


class ADWINBaggingClassifier(BaggingClassifier):
    """ADWIN Bagging classifier.

    ADWIN Bagging [^1] is the online bagging method of Oza and Russell [^2]
    with the addition of the `ADWIN` algorithm as a change detector. If concept
    drift is detected, the worst member of the ensemble (based on the error
    estimation by ADWIN) is replaced by a new (empty) classifier.

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
            max_error_idx = max(
                range(len(self._drift_detectors)),
                key=lambda j: self._drift_detectors[j].estimation,
            )
            self.models[max_error_idx] = copy.deepcopy(self.model)
            self._drift_detectors[max_error_idx] = ADWIN()

        return self


class LeveragingBaggingClassifier(BaggingClassifier):
    """Leveraging Bagging ensemble classifier.

    Leveraging Bagging [^1] is an improvement over the Oza Bagging algorithm.
    The bagging performance is leveraged by increasing the re-sampling.
    It uses a poisson distribution to simulate the re-sampling process.
    To increase re-sampling it uses a higher `w` value of the Poisson
    distribution (agerage number of events), 6 by default, increasing the
    input space diversity, by attributing a different range of weights to the
    data samples.

    To deal with concept drift, Leveraging Bagging uses the ADWIN algorithm to
    monitor the performance of each member of the enemble If concept drift is
    detected, the worst member of the ensemble (based on the error estimation
    by ADWIN) is replaced by a new (empty) classifier.

    Parameters
    ----------
    model
        The classifier to bag.
    n_models
        The number of models in the ensemble.
    w
        Indicates the average number of events. This is the lambda parameter
        of the Poisson distribution used to compute the re-sampling weight.
    adwin_delta
        The delta parameter for the ADWIN change detector.
    bagging_method
        The bagging method to use. Can be one of the following:<br/>
        * 'bag' - Leveraging Bagging using ADWIN.<br/>
        * 'me' - Assigns $weight=1$ if sample is misclassified,
          otherwise $weight=error/(1-error)$.<br/>
        * 'half' - Use resampling without replacement for half of the instances.<br/>
        * 'wt' - Resample without taking out all instances.<br/>
        * 'subag' - Resampling without replacement.<br/>
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = ensemble.LeveragingBaggingClassifier(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.886282

    """

    _BAGGING_METHODS = ("bag", "me", "half", "wt", "subag")

    def __init__(
        self,
        model: base.Classifier,
        n_models: int = 10,
        w: float = 6,
        adwin_delta: float = 0.002,
        bagging_method: str = "bag",
        seed: int = None,
    ):
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.n_detected_changes = 0
        self.w = w
        self.adwin_delta = adwin_delta
        self.bagging_method = bagging_method
        self._drift_detectors = [
            copy.deepcopy(ADWIN(delta=self.adwin_delta)) for _ in range(self.n_models)
        ]

        # Set bagging function
        if bagging_method == "bag":
            self._bagging_fct = self._leveraging_bag
        elif bagging_method == "me":
            self._bagging_fct = self._leveraging_bag_me
        elif bagging_method == "half":
            self._bagging_fct = self._leveraging_bag_half
        elif bagging_method == "wt":
            self._bagging_fct = self._leveraging_bag_wt
        elif bagging_method == "subag":
            self._bagging_fct = self._leveraging_subag
        else:
            raise ValueError(
                f"Invalid bagging_method: {bagging_method}\n"
                f"Valid options: {self._BAGGING_METHODS}"
            )

    def _leveraging_bag(self, **kwargs):
        # Leveraging bagging
        return self._rng.poisson(self.w)

    def _leveraging_bag_me(self, **kwargs):
        # Miss-classification error using weight=1 if misclassified.
        # Otherwise using weight=error/(1-error)
        x = kwargs["x"]
        y = kwargs["y"]
        i = kwargs["model_idx"]
        error = self._drift_detectors[i].estimation
        y_pred = self.models[i].predict_one(x)
        if y_pred != y:
            k = 1
        elif error != 1.0 and self._rng.rand() < (error / (1.0 - error)):
            k = 1
        else:
            k = 0
        return k

    def _leveraging_bag_half(self, **kwargs):
        # Resampling without replacement for half of the instances
        return int(not self._rng.randint(2))

    def _leveraging_bag_wt(self, **kwargs):
        # Resampling without taking out all instances
        return 1 + self._rng.poisson(1.0)

    def _leveraging_subag(self, **kwargs):
        # Subagging using resampling without replacement
        return int(self._rng.poisson(1) > 0)

    def learn_one(self, x, y):
        change_detected = False
        for i, model in enumerate(self):
            k = self._bagging_fct(x=x, y=y, model_idx=i)

            for _ in range(k):
                model.learn_one(x, y)

            y_pred = self.models[i].predict_one(x)
            if y_pred is not None:
                incorrectly_classifies = int(y_pred != y)
                error = self._drift_detectors[i].estimation
                self._drift_detectors[i].update(incorrectly_classifies)
                if self._drift_detectors[i].change_detected:
                    if self._drift_detectors[i].estimation > error:
                        change_detected = True

        if change_detected:
            self.n_detected_changes += 1
            max_error_idx = max(
                range(len(self._drift_detectors)),
                key=lambda j: self._drift_detectors[j].estimation,
            )
            self.models[max_error_idx] = copy.deepcopy(self.model)
            self._drift_detectors[max_error_idx] = ADWIN(delta=self.adwin_delta)

        return self

    @property
    def bagging_methods(self):
        """Valid bagging_method options."""
        return self._BAGGING_METHODS
