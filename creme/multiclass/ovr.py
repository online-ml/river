import collections
import copy

from creme import base
from creme import utils


__all__ = ['OneVsRestClassifier']


class OneVsRestClassifier(base.MultiClassifier):
    """One-vs-the-rest (OvR) multiclass strategy.

    This strategy consists in fitting one binary classifier per class. Because we are in a
    streaming context, the number of classes isn't known from the start, hence new classifiers are
    instantiated on the fly. Likewise the predicted probabilities will only include the classes
    seen up to a given point in time.

    Parameters:
        binary_classifier

    Attributes:
        classifiers (dict): A mapping between classes and classifiers.

    Example:

        >>> from creme import compose
        >>> from creme import datasets
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import multiclass
        >>> from creme import optim
        >>> from creme import preprocessing

        >>> X_y = datasets.ImageSegments()

        >>> model = compose.Pipeline(
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', multiclass.OneVsRestClassifier(
        ...         binary_classifier=linear_model.LogisticRegression())
        ...     )
        ... )

        >>> metric = metrics.MacroF1()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        MacroF1: 0.774148

    """

    def __init__(self, binary_classifier: base.BinaryClassifier):
        self.models = {}
        self.binary_classifier = binary_classifier

    def __str__(self):
        return f'OneVsRestClassifier({self.binary_classifier})'

    def fit_one(self, x, y):

        # Instantiate a new binary classifier if the class is new
        if y not in self.models:
            self.models[y] = copy.deepcopy(self.binary_classifier)

        # Train each label's associated classifier
        for label, model in self.models.items():
            model.fit_one(x, y == label)

        return self

    def predict_proba_one(self, x):
        y_pred = {
            label: model.predict_proba_one(x)[True]
            for label, model in self.models.items()
        }
        return utils.math.softmax(y_pred)
