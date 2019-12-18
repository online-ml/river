import collections
import random

from .. import base


class RandomUnderSampler(base.Wrapper, base.Classifier):
    """Random under-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by under-sampling the
    stream of given observations so that the class distribution seen by the classifier ressembles
    a given desired distribution.

    Parameters:
        classifier (base.Classifier)
        desired_dist (dict): The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values must sum up to 1.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import imblearn
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import preprocessing

            >>> X_y = datasets.CreditCard()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     imblearn.RandomUnderSampler(
            ...         classifier=linear_model.LogisticRegression(),
            ...         desired_dist={0: .5, 1: .5},
            ...         seed=42
            ...     )
            ... )

            >>> metric = metrics.ROCAUC()

            >>> model_selection.progressive_val_score(X_y.take(10000), model, metric)
            ROCAUC: 0.945842

    References:
        1. `Under-sampling a dataset with desired ratios <https://maxhalford.github.io/blog/under-sampling-a-dataset-with-desired-ratios/>`_

    """

    def __init__(self, classifier, desired_dist, seed=None):
        self.classifier = classifier
        self.desired_dist = desired_dist
        self.seed = seed
        self._rng = random.Random(seed)
        self._actual_dist = collections.Counter()
        self._pivot = None

    @property
    def _model(self):
        return self.classifier

    def fit_one(self, x, y):

        self._actual_dist[y] += 1

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(
                self._actual_dist.keys(),
                key=lambda x: self.desired_dist[x] / self._actual_dist[x]
            )
        else:
            self.classifier.fit_one(x, y)
            return self

        # Determine the sampling ratio if the class is not the pivot
        ratio = (
            (self.desired_dist[y] * self._actual_dist[self._pivot]) /
            (self.desired_dist[self._pivot] * self._actual_dist[y])
        )

        if self._rng.random() < ratio:
            self.classifier.fit_one(x, y)

        return self

    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x)

    def predict_one(self, x):
        return self.classifier.predict_one(x)
