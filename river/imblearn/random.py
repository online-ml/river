import collections

import numpy as np

from river import base


class ClassificationSampler(base.WrapperMixin, base.Classifier):
    def __init__(self, classifier, seed=None):
        self.classifier = classifier
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _wrapped_model(self):
        return self.classifier

    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x)

    def predict_one(self, x):
        return self.classifier.predict_one(x)


class RandomUnderSampler(ClassificationSampler):
    """Random under-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by under-sampling the
    stream of given observations so that the class distribution seen by the classifier follows
    a given desired distribution. The implementation is a discrete version of rejection sampling.

    Parameters
    ----------
    classifier
    desired_dist
        The desired class distribution. The keys are the classes whilst the values are the desired
        class percentages. The values must sum up to 1.
    seed
        Random seed for reproducibility.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

    References
    ----------
    [^1]: [Under-sampling a dataset with desired ratios](https://maxhalford.github.io/blog/under-sampling-a-dataset-with-desired-ratios/)
    [^2]: [Wikipedia article on rejection sampling](https://www.wikiwand.com/en/Rejection_sampling)

    """

    def __init__(self, classifier: base.Classifier, desired_dist: dict, seed: int = None):
        super().__init__(classifier=classifier, seed=seed)
        self.desired_dist = desired_dist
        self._actual_dist = collections.Counter()
        self._pivot = None

    def learn_one(self, x, y):

        self._actual_dist[y] += 1
        f = self.desired_dist
        g = self._actual_dist

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
        else:
            self.classifier.learn_one(x, y)
            return self

        # Determine the sampling ratio if the class is not the pivot
        M = f[self._pivot] / g[self._pivot]  # Likelihood ratio
        ratio = f[y] / (M * g[y])

        if ratio < 1 and self._rng.random() < ratio:
            self.classifier.learn_one(x, y)

        return self


class RandomOverSampler(ClassificationSampler):
    """Random over-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by over-sampling the
    stream of given observations so that the class distribution seen by the classifier follows
    a given desired distribution. The implementation is a discrete version of reverse rejection
    sampling.

    Parameters
    ----------
    classifier
    desired_dist
        The desired class distribution. The keys are the classes whilst the values are the desired
        class percentages. The values must sum up to 1.
    seed
        Random seed for reproducibility.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

    """

    def __init__(self, classifier: base.Classifier, desired_dist: dict, seed: int = None):
        super().__init__(classifier=classifier, seed=seed)
        self.desired_dist = desired_dist
        self._actual_dist = collections.Counter()
        self._pivot = None

    def learn_one(self, x, y):

        self._actual_dist[y] += 1
        f = self.desired_dist
        g = self._actual_dist

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(g.keys(), key=lambda y: g[y] / f[y])
        else:
            self.classifier.learn_one(x, y)
            return self

        M = g[self._pivot] / f[self._pivot]
        rate = M * f[y] / g[y]

        for _ in range(self._rng.poisson(rate)):
            self.classifier.learn_one(x, y)

        return self


class RandomSampler(ClassificationSampler):
    """Random sampling by mixing under-sampling and over-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by both under-sampling
    and over-sampling the stream of given observations so that the class distribution seen by the
    classifier follows a given desired distribution.

    Parameters
    ----------
    classifier
    desired_dist
        The desired class distribution. The keys are the classes whilst the values are the desired
        class percentages. The values must sum up to 1. If set to `None`, then the observations
        will be sampled uniformly at random, which is stricly equivalent to using
        `ensemble.BaggingClassifier`.
    sampling_rate
        The desired ratio of data to sample.
    seed
        Random seed for reproducibility.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

    """

    def __init__(
        self,
        classifier: base.Classifier,
        desired_dist: dict,
        sampling_rate=1.0,
        seed: int = None,
    ):
        super().__init__(classifier=classifier, seed=seed)
        self.sampling_rate = sampling_rate
        self._actual_dist = collections.Counter()
        if desired_dist is None:
            desired_dist = self._actual_dist
        self.desired_dist = desired_dist
        self._n = 0

    def learn_one(self, x, y):

        self._actual_dist[y] += 1
        self._n += 1
        f = self.desired_dist
        g = self._actual_dist

        rate = self.sampling_rate * f[y] / (g[y] / self._n)

        for _ in range(self._rng.poisson(rate)):
            self.classifier.learn_one(x, y)

        return self
