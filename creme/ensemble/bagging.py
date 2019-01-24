import collections
import copy

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

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when one using a single logistic regression.

        >>> import creme.ensemble
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.optim
        >>> import creme.pipeline
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimiser = creme.optim.VanillaSGD()
        >>> model = creme.pipeline.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.linear_model.LogisticRegression(optimiser))
        ... ])
        >>> model = creme.ensemble.BaggingClassifier(model, n_estimators=3)
        >>> metric = metrics.roc_auc_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.991563...

    - [Online Bagging and Boosting](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, base_estimator=None, n_estimators=10, random_state=42):
        self.base_estimator = base_estimator
        '''The estimator to bag, should implement `creme.base.Classifier`.'''
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
        return sum(estimator.predict_proba_one(x) for estimator in self.estimators) / len(self.estimators)
