from sklearn import pipeline
from sklearn.utils import metaestimators


__all__ = ['Pipeline']


class Pipeline(pipeline.Pipeline):
    """A sequence of estimators.

    During training each observation is processed in the order in which the steps have been
    provided. Each observation is processed independently from the others, which means the whole
    process can act as true producer/consumer pipeline where each estimator can run in parallel
    with the others. Complex pipelines can be built by using `creme.compose.TransformerUnion`s as
    steps.

    `creme`'s `Pipeline` inherits from `sklearn`'s `Pipeline` so it is fully compatible with it.

    """

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def fit_one(self, x, y):
        for _, step in self.steps:
            x = step.fit_one(x, y)
        return x

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def transform_one(self, x):
        for _, step in self.steps:
            x = step.transform_one(x)
        return x

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict_one(self, x):
        for _, step in self.steps[:-1]:
            x = step.transform_one(x)
        return self.steps[-1][1].predict_one(x)

    @metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict_proba_one(self, x):
        for _, step in self.steps[:-1]:
            x = step.transform_one(x)
        return self.steps[-1][1].predict_proba_one(x)
