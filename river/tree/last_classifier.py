from __future__ import annotations

from river import base, drift

from .hoeffding_tree import HoeffdingTree
from .nodes.branch import DTBranch
from .nodes.last_nodes import LeafMajorityClassWithDetector, LeafNaiveBayesWithDetector, LeafNaiveBayesAdaptiveWithDetector
from .nodes.leaf import HTLeaf
from .split_criterion import GiniSplitCriterion, HellingerDistanceCriterion, InfoGainSplitCriterion
from .splitter import GaussianSplitter, Splitter


class LASTClassifier(HoeffdingTree, base.Classifier):
    """Local Adaptive Streaming Tree classifier.

    Parameters
    ----------
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    change_detector
        Change detector that will be created at each leaf of the tree.
    track_error
        If True, the change detector will have binary inputs for error predictions, 
        otherwise the input will be the split criteria.
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
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
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.

    Notes
    -----
    Local Adaptive Streaming Tree [^1] (LAST) is an incremental decision tree with
    adaptive splitting mechanisms. At each leaf, LAST maintains a change detector, 
    that in case of a change detection in error or the data distribution of the leaf,
    it performs a split.

    
    

    References
    ----------

    [^1]: Daniel Nowak Assis, Jean Paul Barddal, and Fabrício Enembreck.
    Just Change on Change: Adaptive Splitting Time for Decision Trees in
    Data Stream Classification . In Proceedings of ACM SAC Conference (SAC’24).

    Examples
    --------

    >>> from river.datasets import synth
    >>> from river.datasets import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
    ...                        drift_stream=synth.SEA(seed=42, variant=1),
    ...                        seed=1, position=500, width=50)
    Accuracy: 91.60%
    """

    _GINI_SPLIT = "gini"
    _INFO_GAIN_SPLIT = "info_gain"
    _HELLINGER_SPLIT = "hellinger"
    _VALID_SPLIT_CRITERIA = [_GINI_SPLIT, _INFO_GAIN_SPLIT, _HELLINGER_SPLIT]

    _MAJORITY_CLASS = "mc"
    _NAIVE_BAYES = "nb"
    _NAIVE_BAYES_ADAPTIVE = "nba"
    _VALID_LEAF_PREDICTION = [_MAJORITY_CLASS, _NAIVE_BAYES, _NAIVE_BAYES_ADAPTIVE]

    def __init__(
        self,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        leaf_prediction: str = "nba",
        change_detector:base.DriftDetector = drift.ADWIN(),
        track_error : bool = True,
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.split_criterion = split_criterion
        self.change_detector = change_detector
        self.track_error = track_error
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:
            self.splitter = GaussianSplitter()
        else:
            if not splitter.is_target_class:
                raise ValueError("The chosen splitter cannot be used in classification tasks.")
            self.splitter = splitter  # type: ignore

        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split

        # To keep track of the observed classes
        self.classes: set = set()

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau"}

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        if split_criterion not in self._VALID_SPLIT_CRITERIA:
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, self._INFO_GAIN_SPLIT
                )
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                "Invalid leaf_prediction option {}', will use default '{}'".format(
                    leaf_prediction, self._NAIVE_BAYES_ADAPTIVE
                )
            )
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if not self.track_error:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return LeafMajorityClassWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone())
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return LeafNaiveBayesWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone())
            else:  # Naives Bayes Adaptive (default)
                return LeafNaiveBayesAdaptiveWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone())
        else:
            split_criterion = self._new_split_criterion()
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return LeafMajorityClassWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone(), split_criterion)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return LeafNaiveBayesWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone(), split_criterion)
            else:  # Naives Bayes Adaptive (default)
                return LeafNaiveBayesAdaptiveWithDetector(initial_stats, depth, self.splitter, self.change_detector.clone(), split_criterion)
    

    def _new_split_criterion(self):
        if self._split_criterion == self._GINI_SPLIT:
            split_criterion = GiniSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._HELLINGER_SPLIT:
            if not self.track_error:
                raise ValueError("The Heillinger distance cannot estimate the purity of a single distribution.\
                                 Use another split criterion or set track_error to True")
            split_criterion = HellingerDistanceCriterion(self.min_branch_fraction)
        else:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)

        return split_criterion

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch, parent_branch: int, **kwargs):
        """Attempt to split a leaf.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node (branch node).
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attributes. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The leaf to evaluate.
        parent
            The leaf's parent.
        parent_branch
            Parent leaf's branch index.
        kwargs
            Other parameters passed to the new branch.
        """
        if not leaf.observed_class_distribution_is_pure():  # type: ignore
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                best_suggestion = max(best_split_suggestions)
                should_split = best_suggestion.merit > 0.0
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = max(best_split_suggestions)
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                self._enforce_size_limit()

    def learn_one(self, x, y, *, w=1.0):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        w
            Sample weight.

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.
        """

        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, HTLeaf):
            node.learn_one(x, y, w=w, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    if node.change_detector.drift_detected:
                        p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, HTLeaf):
                    break
            # Learn from the sample
            node.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    @property
    def _multiclass(self):
        return True
