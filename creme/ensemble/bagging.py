import collections
import copy
import statistics

from sklearn import utils

from .. import base


__all__ = ['BaggingClassifier']


class BaggingClassifier(base.BinaryClassifier):
    """Bagging for classification.

    For each incoming observation, each model's `fit_one` method is called `k` times where `k`
    is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of being equal
    to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6% chance of being
    equal to 3, a 1% chance of being equal to 4, etc. You can do `scipy.stats.poisson(1).pmf(k)`
    for more detailed values.

    Parameters:
        base_estimator (creme.base.Classifier): The estimator to bag.

    Example:

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.

    ::

        >>> from creme import compose
        >>> from creme import ensemble
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import optim
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = optim.VanillaSGD()
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', linear_model.LogisticRegression(optimiser))
        ... ])
        >>> model = ensemble.BaggingClassifier(model, n_estimators=3)
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.967376

    References:

    1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def __init__(self, base_estimator=None, n_estimators=10, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = [copy.deepcopy(base_estimator) for _ in range(n_estimators)]
        self.rng = utils.check_random_state(random_state)

    def fit_one(self, x, y):

        y_pred = self.predict_proba_one(x)

        for estimator in self.estimators:
            for _ in range(self.rng.poisson(1)):
                estimator.fit_one(x, y)

        return y_pred

    def predict_one(self, x):
        votes = collections.Counter((estimator.predict_one(x) for estimator in self.estimators))
        return max(votes, key=votes.get)

    def predict_proba_one(self, x):
        return statistics.mean(
            estimator.predict_proba_one(x)[True]
            for estimator in self.estimators
        )
