import collections

from sklearn import pipeline
from sklearn.utils import metaestimators

from . import base


__all__ = ['Pipeline']


class Pipeline(pipeline.Pipeline):

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


class FeatureUnion(base.Transformer):
    """
    Example
    -------

        #!python
        >>> import creme.feature_extraction
        >>> import creme.pipeline

        >>> X = [
        ...     {'place': 'Taco Bell', 'revenue': 42},
        ...     {'place': 'Burger King', 'revenue': 16},
        ...     {'place': 'Burger King', 'revenue': 24},
        ...     {'place': 'Taco Bell', 'revenue': 58},
        ...     {'place': 'Burger King', 'revenue': 20},
        ...     {'place': 'Taco Bell', 'revenue': 50}
        ... ]

        >>> mean = creme.feature_extraction.GroupBy(
        ...     on='revenue',
        ...     by='place',
        ...     how=creme.stats.Mean()
        ... )
        >>> count = creme.feature_extraction.GroupBy(
        ...     on='revenue',
        ...     by='place',
        ...     how=creme.stats.Count()
        ... )
        >>> agg = creme.pipeline.FeatureUnion([mean, count])

        >>> for x in X:
        ...     print(sorted(agg.fit_one(x).items()))
        [('revenue_count_by_place', 1), ('revenue_mean_by_place', 42.0)]
        [('revenue_count_by_place', 1), ('revenue_mean_by_place', 16.0)]
        [('revenue_count_by_place', 2), ('revenue_mean_by_place', 20.0)]
        [('revenue_count_by_place', 2), ('revenue_mean_by_place', 50.0)]
        [('revenue_count_by_place', 3), ('revenue_mean_by_place', 20.0)]
        [('revenue_count_by_place', 3), ('revenue_mean_by_place', 50.0)]

    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit_one(self, x, y=None):
        return dict(collections.ChainMap(*(t.fit_one(x, y) for t in self.transformers)))

    def transform_one(self, x):
        return dict(collections.ChainMap(*(t.transform_one(x) for t in self.transformers)))
