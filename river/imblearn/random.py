from __future__ import annotations

import collections
import random
import typing

from river import base, utils


class ClassificationSampler(base.Wrapper, base.Classifier):
    def __init__(self, classifier, seed: int | None = None):
        self.classifier = classifier
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def _wrapped_model(self):
        return self.classifier

    def predict_proba_one(self, x, **kwargs):
        return self.classifier.predict_proba_one(x, **kwargs)

    def predict_one(self, x, **kwargs):
        return self.classifier.predict_one(x, **kwargs)


class RandomUnderSampler(ClassificationSampler):
    """Random under-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by under-sampling the
    stream of given observations so that the class distribution seen by the classifier follows
    a given desired distribution. The implementation is a discrete version of rejection sampling.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

    Parameters
    ----------
    classifier
    desired_dist
        The desired class distribution. The keys are the classes whilst the values are the desired
        class percentages. The values must sum up to 1.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = imblearn.RandomUnderSampler(
    ...     (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     desired_dist={False: 0.4, True: 0.6},
    ...     seed=42
    ... )

    >>> dataset = datasets.CreditCard().take(3000)

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.0336...

    References
    ----------
    [^1]: [Under-sampling a dataset with desired ratios](https://maxhalford.github.io/blog/undersampling-ratios/)
    [^2]: [Wikipedia article on rejection sampling](https://www.wikiwand.com/en/Rejection_sampling)

    """

    def __init__(self, classifier: base.Classifier, desired_dist: dict, seed: int | None = None):
        super().__init__(classifier=classifier, seed=seed)
        self.desired_dist = desired_dist
        self._actual_dist: typing.Counter = collections.Counter()
        self._pivot = None

    def learn_one(self, x, y, **kwargs):
        self._actual_dist[y] += 1
        f = self.desired_dist
        g = self._actual_dist

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
        else:
            self.classifier.learn_one(x, y, **kwargs)
            return self

        # Determine the sampling ratio if the class is not the pivot
        M = f[self._pivot] / g[self._pivot]  # Likelihood ratio
        ratio = f[y] / (M * g[y])

        if ratio < 1 and self._rng.random() < ratio:
            self.classifier.learn_one(x, y, **kwargs)

        return self


class RandomOverSampler(ClassificationSampler):
    """Random over-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by over-sampling the
    stream of given observations so that the class distribution seen by the classifier follows
    a given desired distribution. The implementation is a discrete version of reverse rejection
    sampling.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

    Parameters
    ----------
    classifier
    desired_dist
        The desired class distribution. The keys are the classes whilst the values are the desired
        class percentages. The values must sum up to 1.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = imblearn.RandomOverSampler(
    ...     (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     desired_dist={False: 0.4, True: 0.6},
    ...     seed=42
    ... )

    >>> dataset = datasets.CreditCard().take(3000)

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.0457...

    """

    def __init__(self, classifier: base.Classifier, desired_dist: dict, seed: int | None = None):
        super().__init__(classifier=classifier, seed=seed)
        self.desired_dist = desired_dist
        self._actual_dist: typing.Counter = collections.Counter()
        self._pivot = None

    def learn_one(self, x, y, **kwargs):
        self._actual_dist[y] += 1
        f = self.desired_dist
        g = self._actual_dist

        # Check if the pivot needs to be changed
        if y != self._pivot:
            self._pivot = max(g.keys(), key=lambda y: g[y] / f[y])
        else:
            self.classifier.learn_one(x, y, **kwargs)
            return self

        M = g[self._pivot] / f[self._pivot]
        rate = M * f[y] / g[y]

        for _ in range(utils.random.poisson(rate, rng=self._rng)):
            self.classifier.learn_one(x, y, **kwargs)

        return self


class RandomSampler(ClassificationSampler):
    """Random sampling by mixing under-sampling and over-sampling.

    This is a wrapper for classifiers. It will train the provided classifier by both under-sampling
    and over-sampling the stream of given observations so that the class distribution seen by the
    classifier follows a given desired distribution.

    See [Working with imbalanced data](/user-guide/imbalanced-learning) for example usage.

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

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = imblearn.RandomSampler(
    ...     (
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LogisticRegression()
    ...     ),
    ...     desired_dist={False: 0.4, True: 0.6},
    ...     sampling_rate=0.8,
    ...     seed=42
    ... )

    >>> dataset = datasets.CreditCard().take(3000)

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.09...

    """

    def __init__(
        self,
        classifier: base.Classifier,
        desired_dist: dict,
        sampling_rate=1.0,
        seed: int | None = None,
    ):
        super().__init__(classifier=classifier, seed=seed)
        self.sampling_rate = sampling_rate
        self._actual_dist: typing.Counter = collections.Counter()
        if desired_dist is None:
            desired_dist = self._actual_dist
        self.desired_dist = desired_dist
        self._n = 0

    def learn_one(self, x, y, **kwargs):
        self._actual_dist[y] += 1
        self._n += 1
        f = self.desired_dist
        g = self._actual_dist

        rate = self.sampling_rate * f[y] / (g[y] / self._n)

        for _ in range(utils.random.poisson(rate, rng=self._rng)):
            self.classifier.learn_one(x, y, **kwargs)

        return self
