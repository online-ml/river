from ... import ensemble

from . import tree


__all__ = ['RandomForestClassifier']


class RandomForestClassifier(ensemble.BaggingClassifier):
    """Random forest classifier.

    This is a thin wrapper over `ensemble.BaggingClassifier` and `tree.DecisionTreeClassifier`.

    Parameters:
        n_trees (int): The number of trees in the forest.
        seed (int): Random number generator seed for reproducibility.
        **tree_params (dict): The parameters of the decision tree.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import tree

            >>> X_y = datasets.Phishing()

            >>> model = tree.RandomForestClassifier(
            ...     n_trees=10,
            ...     seed=42,
            ...     # Tree parameters
            ...     patience=100,
            ...     confidence=1e-5,
            ...     criterion='gini'
            ... )

            >>> metric = metrics.LogLoss()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            LogLoss: 0.456533

    """

    def __init__(self, n_trees=10, seed=None, **tree_params):
        super().__init__(
            model=tree.DecisionTreeClassifier(**tree_params),
            n_models=n_trees,
            seed=seed
        )
        self.tree_params = tree_params

    @property
    def n_trees(self):
        return len(self)

    def __str__(self):
        return 'RandomForestClassifier'
