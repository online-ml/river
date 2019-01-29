import copy

from .. import base


__all__ = ['OneVsRestClassifier']


class OneVsRestClassifier(base.MultiClassifier):
    """One-vs-the-rest (OvR) multiclass strategy.

    This strategy consists in fitting one binary classifier per class. Because we are in a
    streaming context, the number of classes isn't known from the start, hence it new classifiers
    are created on the fly. Likewise the predicted probabilities will only include the classes seen
    up to now.

    Parameters:
        base_estimator (base.BinaryClassifier)

    Attributes:
        estimators (dict): A mapping between classes and estimators.

    Example:

        >>> import creme.compose
        >>> import creme.linear_model
        >>> import creme.model_selection
        >>> import creme.multiclass
        >>> import creme.optim
        >>> import creme.preprocessing
        >>> import creme.stream
        >>> from sklearn import datasets
        >>> from sklearn import metrics

        >>> X_y = creme.stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_iris,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> optimr = creme.optim.RMSProp()
        >>> model = creme.compose.Pipeline([
        ...     ('scale', creme.preprocessing.StandardScaler()),
        ...     ('learn', creme.multiclass.OneVsRestClassifier(
        ...         base_estimator=creme.linear_model.LogisticRegression(optimr))
        ...     )
        ... ])
        >>> metric = metrics.accuracy_score

        >>> creme.model_selection.online_score(X_y, model, metric)
        0.813333...

    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimators = {}

    def _normalize_preds(self, y):
        ys = sum(y.values())
        return {c: p / ys for c, p in y.items()}

    def fit_one(self, x, y):
        if y not in self.estimators:
            self.estimators[y] = copy.deepcopy(self.base_estimator)
        y_pred = {c: model.fit_one(x, y == c) for c, model in self.estimators.items()}
        return self._normalize_preds(y_pred)

    def predict_proba_one(self, x):
        y_pred = {c: model.predict_proba_one(x) for c, model in self.estimators.items()}
        return self._normalize_preds(y_pred)

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        return max(y_pred, key=y_pred.get)
