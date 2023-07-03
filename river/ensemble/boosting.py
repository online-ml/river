from __future__ import annotations

import collections
import math

from river import base, drift, linear_model, utils

__all__ = ["AdaBoostClassifier", "ADWINBoostingClassifier", "BOLEClassifier"]


class AdaBoostClassifier(base.WrapperEnsemble, base.Classifier):
    """Boosting for classification.

    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter lambda. The lambda parameter is
    updated when the weaks learners fit successively the same observation.

    Parameters
    ----------
    model
        The classifier to boost.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    In the following example three tree classifiers are boosted together. The performance is
    slightly better than when using a single tree.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()

    >>> metric = metrics.LogLoss()

    >>> model = ensemble.AdaBoostClassifier(
    ...     model=(
    ...         tree.HoeffdingTreeClassifier(
    ...             split_criterion='gini',
    ...             delta=1e-5,
    ...             grace_period=2000
    ...         )
    ...     ),
    ...     n_models=5,
    ...     seed=42
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.370805

    >>> print(model)
    AdaBoostClassifier(HoeffdingTreeClassifier)

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int | None = None):
        super().__init__(model, n_models, seed)
        self.wrong_weight: collections.defaultdict = collections.defaultdict(int)
        self.correct_weight: collections.defaultdict = collections.defaultdict(int)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LogisticRegression()}

    def learn_one(self, x, y, **kwargs):
        lambda_poisson = 1

        for i, model in enumerate(self):
            for _ in range(utils.random.poisson(lambda_poisson, self._rng)):
                model.learn_one(x, y, **kwargs)

            if model.predict_one(x) == y:
                self.correct_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.correct_weight[i]
                )

            else:
                self.wrong_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.wrong_weight[i]
                )
        return self

    def predict_proba_one(self, x, **kwargs):
        y_proba = collections.Counter()

        for i, model in enumerate(self):
            epsilon = self.wrong_weight[i] + 1e-16
            epsilon /= (self.correct_weight[i] + self.wrong_weight[i]) + 1e-16
            if epsilon == 0 or epsilon > 0.5:
                model_weight = 1.0
            else:
                beta_inv = (1 - epsilon) / epsilon
                model_weight = math.log(beta_inv) if beta_inv != 0 else 0
            predictions = model.predict_proba_one(x, **kwargs)
            utils.norm.scale_values_in_dict(predictions, model_weight, inplace=True)
            y_proba.update(predictions)

        utils.norm.normalize_values_in_dict(y_proba, inplace=True)
        return y_proba


class ADWINBoostingClassifier(AdaBoostClassifier):
    """ADWIN Boosting classifier.

    ADWIN Boosting [^1] is the online boosting method of Oza and Russell [^2]
    with the addition of the `ADWIN` algorithm as a change detector. If concept
    drift is detected, the worst member of the ensemble (based on the error
    estimation by ADWIN) is replaced by a new (empty) classifier.

    Parameters
    ----------
    model
        The classifier to boost.
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
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> model = ensemble.ADWINBoostingClassifier(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     n_models=3,
    ...     seed=42
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 87.61%

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

    def __init__(self, model: base.Classifier, n_models=10, seed: int | None = None):
        super().__init__(model, n_models, seed)
        self._drift_detectors = [drift.ADWIN() for _ in range(self.n_models)]

    def learn_one(self, x, y, **kwargs):
        change_detected = False
        lambda_poisson = 1.0
        for i, model in enumerate(self):
            for _ in range(utils.random.poisson(1, self._rng)):
                model.learn_one(x, y, **kwargs)

            if model.predict_one(x) == y:
                self.correct_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.correct_weight[i]
                )

            else:
                self.wrong_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.wrong_weight[i]
                )

            y_pred = model.predict_one(x)
            error_estimation = self._drift_detectors[i].estimation
            self._drift_detectors[i].update(int(y_pred == y))
            if self._drift_detectors[i].drift_detected:
                if self._drift_detectors[i].estimation > error_estimation:
                    change_detected = True

        if change_detected:
            max_error_idx = max(
                range(len(self._drift_detectors)),
                key=lambda j: self._drift_detectors[j].estimation,
            )
            self.models[max_error_idx] = self.model.clone()
            self._drift_detectors[max_error_idx] = drift.ADWIN()
            self.correct_weight[max_error_idx] = 0
            self.wrong_weight[max_error_idx] = 0
        return self


class BOLEClassifier(AdaBoostClassifier):
    """Boosting Online Learning Ensemble (BOLE).

    A modified version of Oza Online Boosting Algorithm [^1]. For each incoming observation, each
    model's `learn_one` method is called `k` times where `k` is sampled from a Poisson distribution
    of parameter lambda. The first model to be trained will be the one with worst
    `correct_weight / (correct_weight + wrong_weight)`. The worst model not yet trained will
    receive lambda values for training from the models that incorrectly classified an instance, and
    the best model's not yet trained will receive lambda values for training from the models that
    correctly classified an instance. For more details, see [^2].

    Parameters
    ----------
    model
        The classifier to boost.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.
    error_bound
        Error bound percentage for allowing models to vote.

    Examples
    --------

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import drift
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Elec2().take(3000)

    >>> model = ensemble.BOLEClassifier(
    ...     model=drift.DriftRetrainingClassifier(
    ...         model=tree.HoeffdingTreeClassifier(),
    ...         drift_detector=drift.binary.DDM()
    ...     ),
    ...     n_models=10,
    ...     seed=42
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 93.63%

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
    [^2]: R. S. M. d. Barros, S. Garrido T. de Carvalho Santos and P. M. Gonçalves Júnior, "A Boosting-like Online Learning Ensemble," 2016 International Joint Conference on Neural Networks (IJCNN), 2016, pp. 1871-1878, doi: 10.1109/IJCNN.2016.7727427.

    """

    def __init__(
        self, model: base.Classifier, n_models=10, seed: int | None = None, error_bound=0.5
    ):
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.error_bound = error_bound
        self.order_position = [i for i in range(n_models)]
        self.instances_seen = 0

    def learn_one(self, x, y, **kwargs):
        self.instances_seen += 1

        correct_rate = [0] * self.n_models
        for i in range(self.n_models):
            correct_rate[i] = (
                self.correct_weight[self.order_position[i]]
                + self.wrong_weight[self.order_position[i]]
            )
            if correct_rate[i] != 0:
                correct_rate[i] = self.correct_weight[self.order_position[i]] / correct_rate[i]

        # sort models in descending order by correct_weight / correct_weight + error_weight (best case insertion sort)
        for i in range(1, self.n_models):
            key_position = self.order_position[i]
            key_correct_rate = correct_rate[i]
            j = i - 1
            while (j >= 0) and (correct_rate[j] < key_correct_rate):
                self.order_position[j + 1] = self.order_position[j]
                correct_rate[j + 1] = correct_rate[j]
                j -= 1
            self.order_position[j + 1] = key_position
            correct_rate[j + 1] = key_correct_rate

        correct = False
        max_correct_rate = 0
        min_correct_rate = self.n_models - 1
        lambda_poisson = 1

        # does boosting not in a linear way (i.e a model boosts the model created after him in the ensemble creation process)
        # the first model to be trained will be the one with worst correct_weight / correct_weight + wrong_weight
        # The worst model's not yet trained will receive lambda values for training from the model's that incorrectly classified an instance.
        # the best model's not yet trained will receive lambda values for training from the model's that correctly classified an instance.
        # the values of lambda increase in case a mistake is made and decrease in case a right prediction is made.
        # the worst models are more likely to make mistakes, increasing the value of lambda.
        # Then, the best's model are likely to receive a high value of lambda and decreasing gradually throughout the remaning models to be trained
        # It's similar to a system where the rich get richer.
        for i in range(self.n_models):
            if correct:
                pos = self.order_position[max_correct_rate]
                max_correct_rate += 1
            else:
                pos = self.order_position[min_correct_rate]
                min_correct_rate -= 1

            for _ in range(utils.random.poisson(lambda_poisson, self._rng)):
                self.models[pos].learn_one(x, y, **kwargs)

            if self.models[pos].predict_one(x) == y:
                self.correct_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.correct_weight[pos])
                correct = True
            else:
                self.wrong_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.wrong_weight[pos])
                correct = False
        return self

    def predict_proba_one(self, x, **kwargs):
        y_proba = collections.Counter()
        y_proba_all = (
            collections.Counter()
        )  # stores prediction of every model of the ensemble, if y_proba is null, returns y_proba_all
        for i, model in enumerate(self):
            model_weight = 0.0
            if self.correct_weight[i] > 0.0 and self.wrong_weight[i] > 0.0:
                epsilon = (
                    self.wrong_weight[i] / (self.correct_weight[i] + self.wrong_weight[i]) + 1e-16
                )
                if epsilon <= self.error_bound:
                    beta_inv = (1 - epsilon) / epsilon
                    model_weight = math.log(beta_inv)
            predictions = model.predict_proba_one(x, **kwargs)
            if model_weight:
                utils.norm.scale_values_in_dict(predictions, model_weight, inplace=True)
                y_proba.update(predictions)
            y_proba_all.update(predictions)
        if not len(y_proba):
            utils.norm.normalize_values_in_dict(y_proba_all, inplace=True)
            return y_proba_all
        utils.norm.normalize_values_in_dict(y_proba, inplace=True)
        return y_proba
