from __future__ import annotations

from .hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier
from .split_criterion import GiniSplitCriterion


class ExtremelyFastHoeffdingAdaptiveTreeClassifier(HoeffdingAdaptiveTreeClassifier):
    """Extremely Fast Hoeffding Adaptive Tree classifier.

    Each leaf compares the gain from its best split with keeping the leaf unsplit [^1].
    When a drift detector signals a drop in accuracy, the tree grows an alternate subtree
    and switches to it if its error is significantly lower.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow until
          the system recursion limit.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes. If empty, then assume that all numeric attributes should
        be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    bootstrap_sampling
        If True, perform bootstrap sampling in the leaf nodes.
    drift_window_threshold
        Minimum number of examples an alternate tree must observe before being considered as a
        potential replacement to the current one.
    drift_detector
        The drift detector used to build the tree. If `None` then `drift.ADWIN` is used.
    switch_significance
        The significance level to assess whether alternate subtrees are significantly better
        than their main subtree counterparts.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_size
        The max size of the tree, in mebibytes (MiB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    seed
       Random seed for reproducibility.

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
