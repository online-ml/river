from sklearn.utils import metaestimators


__all__ = ['Pipeline']


class Pipeline:
    """A sequence of estimators.

    During training each observation is processed in the order in which the steps have been
    provided. Each observation is processed independently from the others, which means the whole
    process can act as true producer/consumer pipeline where each estimator can run in parallel
    with the others.

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

    def fit_one(self, x, y=None):
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
