from __future__ import annotations

from .hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier
from .split_criterion import GiniSplitCriterion


class ExtremelyFastHoeffdingAdaptiveTreeClassifier(HoeffdingAdaptiveTreeClassifier):
    """Extremely Fast Hoeffding Adaptive Tree classifier.

    Each leaf compares the gain from its best split with keeping the leaf unsplit [^1].
    When a drift detector signals a drop in accuracy, the tree grows an alternate subtree
    and switches to it if its error is significantly lower.

    Examples
    --------
    >>> from river import tree

    >>> model = tree.ExtremelyFastHoeffdingAdaptiveTreeClassifier(
    ...     grace_period=20, bootstrap_sampling=False
    ... )
    >>> for i in range(20):
    ...     state = "on" if i % 2 else "off"
    ...     model.learn_one({"state": state}, state == "on")
    >>> model.n_branches
    1
    >>> model.predict_one({"state": "on"})
    True

    References
    ----------
    [^1]: Manapragada, C., Salehi, M., and Webb, G. I. (2022).
        [Extremely Fast Hoeffding Adaptive Tree](https://doi.org/10.1109/ICDM54844.2022.00042).
    """

    def _split_comparison_merit(self, suggestions, criterion, stats) -> float | None:
        if not suggestions or suggestions[-1].feature is None:
            return None
        if isinstance(criterion, GiniSplitCriterion):
            return 1.0 - criterion.current_merit(stats)
        return 0.0
