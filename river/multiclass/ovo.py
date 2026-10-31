from __future__ import annotations

import collections
import collections.abc
import copy
import functools
import typing

from river import base, linear_model

__all__ = ["OneVsOneClassifier"]


class OneVsOneClassifier(base.Wrapper[base.Classifier], base.Classifier):
    """One-vs-One (OvO) multiclass strategy.

    This strategy consists in fitting one binary classifier for each pair of classes. Because we
    are in a streaming context, the number of classes isn't known from the start, hence new
    classifiers are instantiated on the fly.

    The number of classifiers is `k * (k - 1) / 2`, where `k` is the number of classes. However,
    each call to `learn_one` only requires training `k - 1` models. Indeed, only the models that
    pertain to the given label have to be trained. Meanwhile, making a prediction requires going
    through each and every model.

    Parameters
    ----------
    classifier
        A binary classifier, although a multi-class classifier will work too.

    Attributes
    ----------
    classifiers : dict
        A mapping between pairs of classes and classifiers. The keys are tuples which contain a
        pair of classes. Each pair is sorted in lexicographical order.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multiclass
    >>> from river import preprocessing

    >>> dataset = datasets.ImageSegments()

    >>> scaler = preprocessing.StandardScaler()
    >>> ovo = multiclass.OneVsOneClassifier(linear_model.LogisticRegression())
    >>> model = scaler | ovo

    >>> metric = metrics.MacroF1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MacroF1: 80.76%

    """

    def __init__(self, classifier: base.Classifier):
        self.classifier = classifier
        new_clf: collections.abc.Callable[[], base.Classifier] = functools.partial(
            copy.deepcopy, classifier
        )
        self.classifiers: collections.defaultdict[
            tuple[base.typing.ClfTarget, base.typing.ClfTarget], base.Classifier
        ] = collections.defaultdict(new_clf)
        self.classes: set[base.typing.ClfTarget] = set()

    @property
    def _wrapped_model(self) -> base.Classifier:
        return self.classifier

    @property
    def _multiclass(self) -> bool:
        return True

    @classmethod
    def _unit_test_params(cls) -> collections.abc.Iterator[dict[str, typing.Any]]:
        yield {"classifier": linear_model.LogisticRegression()}

    def learn_one(
        self,
        x: dict[base.typing.FeatureName, typing.Any],
        y: base.typing.ClfTarget,
        **kwargs: typing.Any,
    ) -> None:
        self.classes.add(y)

        for c in self.classes - {y}:
            is_less = typing.cast(base.typing.SupportsComparisonLesser, c) < y
            pair = (c, y) if is_less else (y, c)
            self.classifiers[pair].learn_one(x, y=is_less, **kwargs)

    def predict_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> base.typing.ClfTarget | None:
        if not self.classifiers:  # is empty
            return None

        votes: collections.defaultdict[base.typing.ClfTarget, int] = collections.defaultdict(int)

        for pair, clf in self.classifiers.items():
            if clf.predict_one(x, **kwargs):
                votes[pair[1]] += 1
            else:
                votes[pair[0]] += 1

        return max(votes, key=votes.__getitem__)
