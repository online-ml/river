import copy

from .. import base
from .. import utils


class OneVsRestClassifier(base.MultiClassifier):
    """One-vs-the-rest (OvR) multiclass strategy.

    This strategy consists in fitting one binary classifier per class. Because we are in a
    streaming context, the number of classes isn't known from the start, hence new classifiers are
    instantiated on the fly. Likewise the predicted probabilities will only include the classes
    seen up to a given point in time.

    Parameters:
        binary_classifier (base.BinaryClassifier)

    Attributes:
        classifiers (dict): A mapping between classes and classifiers.

    Example:

    ::

        >>> from creme import compose
        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import multiclass
        >>> from creme import optim
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_iris,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimizer = optim.RMSProp()
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('learn', multiclass.OneVsRestClassifier(
        ...         binary_classifier=linear_model.LogisticRegression(optimizer))
        ...     )
        ... ])
        >>> metric = metrics.Accuracy()

        >>> model_selection.online_score(X_y, model, metric)
        Accuracy: 0.806667

    """

    def __init__(self, binary_classifier: base.BinaryClassifier):
        self.binary_classifier = binary_classifier
        self.classifiers = {}

    def fit_one(self, x, y):

        # Instantiate a new binary classifier if the class is new
        if y not in self.classifiers:
            self.classifiers[y] = copy.deepcopy(self.binary_classifier)

        # Train each label's associated classifier
        for label, model in self.classifiers.items():
            model.fit_one(x, y == label)

        return self

    def predict_proba_one(self, x):
        y_pred = {
            label: model.predict_proba_one(x)[True]
            for label, model in self.classifiers.items()
        }
        return utils.softmax(y_pred)
