from .. import ensemble

from . import tree


__all__ = ['RandomForestClassifier']


class RandomForestClassifier(ensemble.BaggingClassifier):
    """Random forest classifier.

    This is a thin wrapper over `ensemble.BaggingClassifier` and `tree.DecisionTreeClassifier`.

    Parameters:
        n_trees (int): The number of trees in the forest.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.
        **tree_params (dict): The parameters of the decision tree.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import tree

            >>> X_y = datasets.fetch_electricity()

            >>> model = tree.RandomForestClassifier(
            ...     n_trees=5,
            ...     random_state=42,
            ...     # Tree parameters
            ...     patience=2000,
            ...     confidence=1e-5,
            ...     criterion='gini'
            ... )

            >>> metric = metrics.LogLoss()

            >>> model_selection.online_score(X_y, model, metric)
            LogLoss: 0.529859

    """

    def __init__(self, n_trees=10, random_state=None, **tree_params):
        super().__init__(
            model=tree.DecisionTreeClassifier(**tree_params),
            n_models=n_trees,
            random_state=random_state
        )

    def __str__(self):
        return 'RandomForestClassifier'
