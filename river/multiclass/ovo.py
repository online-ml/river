import collections
import copy
import functools

from river import base
from river import linear_model


__all__ = ["OneVsOneClassifier"]


class OneVsOneClassifier(base.WrapperMixin, base.Classifier):
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
    MacroF1: 0.807573

    """

    def __init__(self, classifier):
        self.classifier = classifier
        new_clf = functools.partial(copy.deepcopy, classifier)
        self.classifiers = collections.defaultdict(new_clf)
        self.classes = set()

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return True

    @classmethod
    def _unit_test_params(cls):
        return {"classifier": linear_model.LogisticRegression()}

    def learn_one(self, x, y):

        self.classes.add(y)

        for c in self.classes - {y}:
            pair = (c, y) if c < y else (y, c)
            self.classifiers[pair].learn_one(x, y=c < y)

        return self

    def predict_one(self, x):

        if not self.classifiers:  # is empty
            return None

        votes = collections.defaultdict(int)

        for pair, clf in self.classifiers.items():
            if clf.predict_one(x):
                votes[pair[1]] += 1
            else:
                votes[pair[0]] += 1

        return max(votes, key=votes.get)
