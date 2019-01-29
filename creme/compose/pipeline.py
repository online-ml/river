from sklearn import pipeline
from sklearn.utils import metaestimators


__all__ = ['Pipeline']


class Pipeline(pipeline.Pipeline):
    """A sequence of estimators.

    During training each observation is processed in the order in which the steps have been
    provided. Each observation is processed independently from the others, which means the whole
    process can act as true producer/consumer pipeline where each estimator can run in parallel
    with the others.

    ``creme``'s ``Pipeline`` inherits from ``sklearn``'s ``Pipeline`` so it is fully compatible
    with it.

    Parameters:
        steps (list): List of (name, transform) tuples (implementing fit/transform) that are
            chained, in the order in which they are chained, with the last object an estimator.

    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def _final_estimator(self):
        """Returns the final estimator."""
        return self.steps[-1][1]

    def fit_one(self, x, y):
        """Fits each steps with ``x``."""
        for _, step in self.steps:
            x = step.fit_one(x, y)
        return x

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def transform_one(self, x):
        """Runs ``x`` through each transformer.

        Only works if the final estimator is a transformer.

        """

        for _, step in self.steps:
            x = step.transform_one(x)
        return x

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict_one(self, x):
        """Predicts the output of ``x``."""
        for _, step in self.steps[:-1]:
            x = step.transform_one(x)
        return self.steps[-1][1].predict_one(x)

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict_proba_one(self, x):
        """Predicts the probability outcome of ``x``.

        Only works if the final estimator is a classifier.

        """
        for _, step in self.steps[:-1]:
            x = step.transform_one(x)
        return self.steps[-1][1].predict_proba_one(x)

    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator."""
        return super().decision_function(X)

    def fit(self, X, y=None, **fit_params):
        """Fit the pipeline to an entire dataset contained in memory."""
        return super().fit(X, y, **fit_params)

    def fit_predict(self, X, y=None):
        """Applies fit_predict of last step in pipeline after transforms."""
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        """Fit the model and transform with the final estimator."""
        return super().fit(X, y)

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order."""
        return super().inverse_transform

    def transform(self, X):
        """Apply transforms, and transform with the final estimator."""
        return super().transform(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return super().get_params(deep)

    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator."""
        return super().predict(X, **predict_params)

    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator."""
        return super().predict_log_proba(X)

    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator."""
        return super().predict_proba(X)

    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and predict_proba of the final estimator."""
        return super().score(X, y, sample_weight)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        return super().set_params(**kwargs)
