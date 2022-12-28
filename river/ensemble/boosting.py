import collections
import copy
import math
import typing

import numpy as np

from river import base, linear_model
from river.base import DriftDetector
from river.drift import DDM
from river.utils.norm import normalize_values_in_dict, scale_values_in_dict
from river.utils.random import poisson

__all__ = ["AdaBoostClassifier", "BOLEClassifier"]


class AdaBoostClassifier(base.WrapperEnsemble, base.Classifier):
    """Boosting for classification

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

    Attributes
    ----------
    wrong_weight : collections.defaultdict
        Number of times a model has made a mistake when making predictions.
    correct_weight : collections.defaultdict
        Number of times a model has predicted the right label when making predictions.

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

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super().__init__(model, n_models, seed)
        self.wrong_weight: typing.DefaultDict = collections.defaultdict(int)
        self.correct_weight: typing.DefaultDict = collections.defaultdict(int)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LogisticRegression()}

    def learn_one(self, x, y):
        lambda_poisson = 1

        for i, model in enumerate(self):
            for _ in range(poisson(lambda_poisson, self._rng)):
                model.learn_one(x, y)

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

    def predict_proba_one(self, x):
        y_proba = collections.Counter()

        for i, model in enumerate(self):
            epsilon = self.wrong_weight[i] + 1e-16
            epsilon /= (self.correct_weight[i] + self.wrong_weight[i]) + 1e-16
            if epsilon == 0 or epsilon > 0.5:
                model_weight = 1.0
            else:
                beta_inv = (1 - epsilon) / epsilon
                model_weight = math.log(beta_inv) if beta_inv != 0 else 0
            predictions = model.predict_proba_one(x)
            normalize_values_in_dict(predictions, inplace=True)
            scale_values_in_dict(predictions, model_weight, inplace=True)
            y_proba.update(predictions)

        normalize_values_in_dict(y_proba, inplace=True)
        return y_proba


class BOLEClassifier(AdaBoostClassifier):
    """Boosting Online Learning Ensemble (BOLE)

    A modified version of Oza Online Boosting Algorithm [^1]. For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter lambda. The first model to be trained will be the one with worst correct_weight / (correct_weight + wrong_weight).
    The worst model's not yet trained will receive lambda values for training from the model's that incorrectly classified an instance, and the best model's not yet trained
    will receive lambda values for training from the model's that correctly classified an instance. For more details, see [^2].

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

    Attributes
    ----------
    wrong_weight : collections.defaultdict
        Number of times a model has made a mistake when making predictions.
    correct_weight : collections.defaultdict
        Number of times a model has predicted the right label when making predictions.
    orderPosition :
        Array with the index of the models with best (correct_weight / correct_weight + wrong_weight) in descending order.
    instances_seen :
        Number of instances that the ensemble trained with.

    Examples
    --------

    >>> from river.ensemble.boosting import BOLEBaseModel
    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import drift
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Elec2()


    >>> model = ensemble.BOLEClassifier(
    ...     model= BOLEBaseModel(model=tree.HoeffdingTreeClassifier(), drift_detection_method= drift.DDM()),
    ...     n_models=10,
    ...     seed=42
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 92.02%



    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    [^2]: R. S. M. d. Barros, S. Garrido T. de Carvalho Santos and P. M. Gonçalves Júnior, "A Boosting-like Online Learning Ensemble," 2016 International Joint Conference on Neural Networks (IJCNN), 2016, pp. 1871-1878, doi: 10.1109/IJCNN.2016.7727427.
    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None, error_bound=0.5):
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.error_bound = error_bound
        self.orderPosition = np.arange(n_models)
        self.instances_seen = 0

    def learn_one(self, x, y):
        self.instances_seen += 1

        acc = np.zeros(self.n_models)
        for i in range(self.n_models):
            acc[i] = (
                self.correct_weight[self.orderPosition[i]]
                + self.wrong_weight[self.orderPosition[i]]
            )
            if acc[i] != 0:
                acc[i] = self.correct_weight[self.orderPosition[i]] / acc[i]

        # sort models in descending order by correct_weight / correct_weight + error_weight (best case insertion sort)
        for i in range(1, self.n_models):
            key_position = self.orderPosition[i]
            key_acc = acc[i]
            j = i - 1
            while (j >= 0) and (acc[j] < key_acc):
                self.orderPosition[j + 1] = self.orderPosition[j]
                acc[j + 1] = acc[j]
                j -= 1
            self.orderPosition[j + 1] = key_position
            acc[j + 1] = key_acc

        correct = False
        maxAcc = 0
        minAcc = self.n_models - 1
        lambda_poisson = 1

        models = list(enumerate(self))
        for i in range(len(models)):
            if correct:
                pos = self.orderPosition[maxAcc]
                maxAcc += 1
            else:
                pos = self.orderPosition[minAcc]
                minAcc -= 1

            for _ in range(poisson(lambda_poisson, self._rng)):
                models[pos][1].learn_one(x, y)

            if models[pos][1].predict_one(x) == y:
                self.correct_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.correct_weight[pos])
                correct = True
            else:
                self.wrong_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.wrong_weight[pos])
                correct = False
        return self

    def predict_proba_one(self, x):
        y_proba = collections.Counter()
        for i, model in enumerate(self):
            model_weight = 0.0
            if self.correct_weight[i] > 0.0 and self.wrong_weight[i] > 0.0:
                epsilon = self.wrong_weight[i] / (self.correct_weight[i] + self.wrong_weight[i])
                if epsilon <= self.error_bound:
                    beta_inv = (1 - epsilon) / epsilon
                    model_weight = np.log(beta_inv)

            if model_weight != 0.0:
                predictions = model.predict_proba_one(x)
                normalize_values_in_dict(predictions, inplace=True)
                scale_values_in_dict(predictions, model_weight, inplace=True)
                y_proba.update(predictions)

        normalize_values_in_dict(y_proba, inplace=True)
        return y_proba


class BOLEBaseModel(base.Classifier):
    """BOLEBaseModel

    In case a warning is detected, a background model starts to train. If a drift is detected, the model will be replaced by the background model.

    Parameters
    ----------
    model
        The classifier and background classifier class.
    drift_detection_method
        Algorithm to track warnings and concept drifts
    """

    def __init__(self, model: base.Classifier, drift_detection_method: DriftDetector):
        self.base_class = model
        self.model = copy.deepcopy(self.base_class)
        self.bkg_model = None
        self.drift_detection_method = (
            drift_detection_method if drift_detection_method is not None else DDM()
        )

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def learn_one(self, x, y):
        self.update_ddm(x, y)
        return self.model.learn_one(x, y)

    def update_ddm(self, x, y):
        y_pred = self.model.predict_one(x)
        if y_pred is not None:
            incorrectly_classifies = int(y_pred != y)
            self.drift_detection_method.update(incorrectly_classifies)
            if self.drift_detection_method.warning_detected:
                if self.bkg_model is None:
                    self.bkg_model = copy.deepcopy(self.base_class)
                self.bkg_model.learn_one(x, y)
            elif self.drift_detection_method._drift_detected:
                if self.bkg_model is not None:
                    self.model = copy.deepcopy(self.bkg_model)
                    self.bkg_model = None
                else:
                    self.model = copy.deepcopy(self.base_class)
