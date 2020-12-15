import collections
import copy
import math

import numpy as np

from river import base
from river import linear_model


__all__ = ["AdaBoostClassifier"]


class AdaBoostClassifier(base.WrapperMixin, base.EnsembleMixin, base.Classifier):
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
    ...             split_confidence=1e-5,
    ...             grace_period=2000
    ...         )
    ...     ),
    ...     n_models=5,
    ...     seed=42
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.37111

    >>> print(model)
    AdaBoostClassifier(HoeffdingTreeClassifier)

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super().__init__(copy.deepcopy(model) for _ in range(n_models))
        self.n_models = n_models
        self.model = model
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.wrong_weight = collections.defaultdict(int)
        self.correct_weight = collections.defaultdict(int)

    @property
    def _wrapped_model(self):
        return self.model

    @classmethod
    def _unit_test_params(cls):
        return {"model": linear_model.LogisticRegression()}

    def learn_one(self, x, y):
        lambda_poisson = 1

        for i, model in enumerate(self):
            for _ in range(self._rng.poisson(lambda_poisson)):
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
        model_weights = collections.defaultdict(int)
        predictions = {}

        for i, model in enumerate(self):
            epsilon = self.correct_weight[i] + 1e-16
            epsilon /= self.wrong_weight[i] + 1e-16
            weight = math.log(epsilon)
            model_weights[i] += weight
            predictions[i] = model.predict_proba_one(x)

        y_pred = predictions[max(model_weights, key=model_weights.get)]
        total = sum(y_pred.values())
        return {label: proba / total for label, proba in y_pred.items()}
