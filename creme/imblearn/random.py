import collections
import random

from .. import base


class RandomUnderSampler(base.Wrapper, base.Classifier):
    """Random under-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by under-sampling the
    stream of given observations so that the class distribution seen by the classifier ressembles
    a given desired distribution. This is in fact a discrete version of rejection sampling.

    Parameters:
        classifier (base.Classifier)
        desired_dist (dict): The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values must sum up to 1.

    See :ref:`Working with imbalanced data` for example usage.

    References:
        1. `Under-sampling a dataset with desired ratios <https://maxhalford.github.io/blog/under-sampling-a-dataset-with-desired-ratios/>`_
        2. `Wikipedia article on rejection sampling <https://www.wikiwand.com/en/Rejection_sampling>`_

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
        f = self.desired_dist
        g = self._actual_dist

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(g.keys(), key=lambda x: f[x] / g[x])
        else:
            self.classifier.fit_one(x, y)
            return self

        # Determine the sampling ratio if the class is not the pivot
        M = f[self._pivot] / g[self._pivot]
        ratio = f[y] / (M * g[y])

        if self._rng.random() < ratio:
            self.classifier.fit_one(x, y)

        return self

    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x)

    def predict_one(self, x):
        return self.classifier.predict_one(x)
