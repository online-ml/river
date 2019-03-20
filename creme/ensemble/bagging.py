import collections
import copy

from sklearn import utils

from .. import base


__all__ = ['BaggingClassifier']


class BaggingClassifier(base.BinaryClassifier, base.MultiClassifier):
    """Bagging for classification.

    For each incoming observation, each model's `fit_one` method is called `k` times where `k`
    is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of being equal
    to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6% chance of being
    equal to 3, a 1% chance of being equal to 4, etc. You can do `scipy.stats.poisson(1).pmf(k)`
    for more detailed values.

    Parameters:
        base_classifier (BinaryClassifier or MultiClassifier): The classifier to bag.

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
        >>> model = ensemble.BaggingClassifier(model, n_classifiers=3)
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.967468

    References:

    1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def __init__(self, base_classifier=None, n_classifiers=10, random_state=42):
        self.base_classifier = base_classifier
        self.classifiers = [copy.deepcopy(base_classifier) for _ in range(n_classifiers)]
        self.rng = utils.check_random_state(random_state)

    @property
    def __class__(self):
        return self.base_classifier.__class__

    def __str__(self):
        return f'BaggingClassifier({str(self.base_classifier)})'

    def fit_one(self, x, y):

        for classifier in self.classifiers:
            for _ in range(self.rng.poisson(1)):
                classifier.fit_one(x, y)

        return self

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.defaultdict(float)

        # Sum the predictions
        for classifier in self.classifiers:
            for label, proba in classifier.predict_proba_one(x).items():
                y_pred[label] += proba

        # Divide by the number of predictions
        for label in y_pred:
            y_pred[label] /= len(self.classifiers)

        return dict(y_pred)
