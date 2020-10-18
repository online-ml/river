from ... import ensemble

from . import tree


__all__ = ['RandomForestClassifier']


class RandomForestClassifier(ensemble.BaggingClassifier):
    """Random forest classifier.

    This is a thin wrapper over `ensemble.BaggingClassifier` and `tree.DecisionTreeClassifier`.

    Parameters
    ----------
    n_trees
        The number of trees in the forest.
    seed
        Random number generator seed for reproducibility.
    tree_params
        The parameters of the decision tree.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()

    >>> model = tree.RandomForestClassifier(
    ...     n_trees=10,
    ...     seed=42,
    ...     # Tree parameters
    ...     patience=100,
    ...     confidence=1e-5,
    ...     criterion='gini'
    ... )

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.456533

    """

    def __init__(self, n_trees=10, seed: int = None, **tree_params: dict):
        super().__init__(
            model=tree.DecisionTreeClassifier(**tree_params),
            n_models=n_trees,
            seed=seed
        )

    @classmethod
    def _default_params(cls):
        return {}

    @property
    def n_trees(self):
        return len(self)

    def _get_params(self):
        return {**super()._get_params(), **self.model._get_params()}

    def __str__(self):
        return 'RandomForestClassifier'
