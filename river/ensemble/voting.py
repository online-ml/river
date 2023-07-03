from __future__ import annotations

import collections

from river import base, linear_model, naive_bayes, tree


class VotingClassifier(base.Classifier, base.Ensemble):
    """Voting classifier.

    A classification is made by aggregating the predictions of each model in the ensemble. The
    probabilities for each class are summed up if `use_probabilities` is set to `True`. If not,
    the probabilities are ignored and each prediction is weighted the same. In this case, it's
    important that you use an odd number of classifiers. A random class will be picked if the
    number of classifiers is even.

    Parameters
    ----------
    models
        The classifiers.
    use_probabilities
        Whether or to weight each prediction with its associated probability.

    Examples
    --------

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import naive_bayes
    >>> from river import preprocessing
    >>> from river import tree

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     ensemble.VotingClassifier([
    ...         linear_model.LogisticRegression(),
    ...         tree.HoeffdingTreeClassifier(),
    ...         naive_bayes.GaussianNB()
    ...     ])
    ... )

    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 86.94%

    """

    def __init__(self, models: list[base.Classifier], use_probabilities=True):
        super().__init__(models)  # type: ignore
        self.use_probabilities = use_probabilities

    @property
    def _multiclass(self):
        return all(model._multiclass for model in self)

    def learn_one(self, x, y):
        for model in self:
            model.learn_one(x, y)
        return self

    def predict_one(self, x):
        if self.use_probabilities:
            votes = (model.predict_proba_one(x) for model in self)
        else:
            votes = ({model.predict_one(x): 1} for model in self)
        agg = collections.Counter()
        for vote in votes:
            agg.update(vote)
        return agg.most_common(1)[0][0] if agg else None

    @classmethod
    def _unit_test_params(cls):
        yield {
            "models": [
                linear_model.LogisticRegression(),
                tree.HoeffdingTreeClassifier(),
                naive_bayes.GaussianNB(),
            ]
        }
