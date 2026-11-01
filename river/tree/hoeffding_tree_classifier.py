from __future__ import annotations

from river import base
from river._river_rust.tree import NativeHoeffdingTreeClassifier

from ._numeric_adapter import NumericAdapter
from .hoeffding_tree import HoeffdingTree
from .nodes.branch import DTBranch, NumericBinaryBranch
from .nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from .nodes.leaf import HTLeaf
from .split_criterion import GiniSplitCriterion, HellingerDistanceCriterion, InfoGainSplitCriterion
from .splitter import GaussianSplitter, Splitter


class HoeffdingTreeClassifier(HoeffdingTree, base.Classifier):
    """Hoeffding Tree or Very Fast Decision Tree classifier.

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
        The max size of the tree, in mebibytes (MiB).
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
    A Hoeffding Tree [^1] is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the distribution generating
    examples does not change over time. Hoeffding trees exploit the fact that a small sample can
    often be enough to choose an optimal splitting attribute. This idea is supported mathematically
    by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of
    an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision
    tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a non-incremental
    learner using infinitely many examples. Implementation based on MOA [^2].

    References
    ----------

    [^1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------

    >>> from river.datasets import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingTreeClassifier(
    ...     grace_period=100,
    ...     delta=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode']
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 84.58%
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
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
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
        native_enabled = (
            self._native_backend_is_compatible()
            and not nominal_attributes
            and (splitter is None or type(splitter) is GaussianSplitter)
            and split_criterion != self._HELLINGER_SPLIT
            and max_size == 100.0
            and self._native_memory_estimate_period_is_compatible(memory_estimate_period)
            and not stop_mem_management
            and not remove_poor_attrs
        )
        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
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
        self._native_enabled = native_enabled
        self._native_model: NativeHoeffdingTreeClassifier | None = None
        self._native_adapter: NumericAdapter | None = None
        self._native_mutable_parameters = (grace_period, delta, tau)
        self._native_random = self._native_rng()

    def _native_backend_is_compatible(self):
        return type(self) is HoeffdingTreeClassifier

    def _native_max_features(self):
        return None

    def _native_rng(self):
        return None

    def _native_sort_features(self):
        return False

    def _native_memory_estimate_period_is_compatible(self, memory_estimate_period):
        return memory_estimate_period == 1_000_000

    def _init_native(self, x, y):
        if not x:
            self._native_enabled = False
            return
        native_seed = 42 if self._native_random is None else self._native_random.getrandbits(64)
        self._native_adapter = NumericAdapter(x, y, sort_features=self._native_sort_features())
        self.classes.add(y)
        self._native_model = NativeHoeffdingTreeClassifier(
            n_features=len(x),
            n_classes=2,
            max_features=self._native_max_features(),
            grace_period=self.grace_period,
            delta=self.delta,
            tau=self.tau,
            n_splits=self.splitter.n_splits,
            max_depth=self.max_depth,
            split_criterion=self.split_criterion,
            leaf_prediction=self.leaf_prediction,
            nb_threshold=self.nb_threshold,
            min_branch_fraction=self.min_branch_fraction,
            max_share_to_split=self.max_share_to_split,
            merit_preprune=self.merit_preprune,
            seed=native_seed,
        )

    def _learn_one_native(self, x, y, w):
        if self._native_model is None:
            self._init_native(x, y)
            if self._native_model is None:
                return False
        adapter = self._native_adapter
        model = self._native_model
        class_index, added_class = adapter.add_label(y)
        if added_class:
            model.insert_class(class_index, len(adapter.classes) > 2)
            self.classes.add(y)
        mutable_parameters = (self.grace_period, self.delta, self.tau)
        if mutable_parameters != self._native_mutable_parameters:
            model.set_mutable_parameters(*mutable_parameters)
            self._native_mutable_parameters = mutable_parameters
        model.learn_one_mapping(x, adapter.features, adapter.feature_names, class_index, w)
        self._train_weight_seen_by_model += w
        return True

    @property
    def height(self):
        if self._native_model is not None:
            return self._native_model.height()
        return super().height

    @property
    def n_nodes(self):
        if self._native_model is not None:
            return self._native_model.n_nodes()
        return super().n_nodes

    @property
    def n_branches(self):
        if self._native_model is not None:
            return self._native_model.n_branches()
        return super().n_branches

    @property
    def n_leaves(self):
        if self._native_model is not None:
            return self._native_model.n_leaves()
        return super().n_leaves

    @property
    def n_active_leaves(self):
        if self._native_model is not None:
            return self._native_model.n_active_leaves()
        return super().n_active_leaves

    @property
    def n_inactive_leaves(self):
        if self._native_model is not None:
            return self._native_model.n_leaves() - self._native_model.n_active_leaves()
        return super().n_inactive_leaves

    @property
    def summary(self):
        if self._native_model is None:
            return super().summary
        return {
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "n_leaves": self.n_leaves,
            "n_active_leaves": self.n_active_leaves,
            "n_inactive_leaves": self.n_inactive_leaves,
            "height": self.height,
            "total_observed_weight": self._native_model.total_weight_seen(),
        }

    def _root_for_inspection(self):
        if self._native_model is None:
            return self._root
        assert self._native_adapter is not None
        records = self._native_model.nodes()
        nodes = [None] * len(records)
        for index in range(len(nodes) - 1, -1, -1):
            feature, threshold, left, right, counts, depth, active = records[index]
            stats = {
                label: counts[class_index]
                for class_index, label in enumerate(self._native_adapter.classes)
                if counts[class_index]
            }
            if feature < 0:
                node = self._new_leaf(stats)
                node.depth = depth
                if not active:
                    node.deactivate()
                nodes[index] = node
            else:
                nodes[index] = NumericBinaryBranch(
                    stats,
                    self._native_adapter.feature_names[feature],
                    threshold,
                    depth,
                    nodes[left],
                    nodes[right],
                )
        return nodes[0]

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau"}

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        if split_criterion not in self._VALID_SPLIT_CRITERIA:
            print(
                f"Invalid split_criterion option {split_criterion}', will use default '{self._INFO_GAIN_SPLIT}'"
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                f"Invalid leaf_prediction option {leaf_prediction}', will use default '{self._NAIVE_BAYES_ADAPTIVE}'"
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

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LeafMajorityClass(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LeafNaiveBayes(initial_stats, depth, self.splitter)
        else:  # Naives Bayes Adaptive (default)
            return LeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)

    def _new_split_criterion(self):
        if self._split_criterion == self._GINI_SPLIT:
            split_criterion = GiniSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._HELLINGER_SPLIT:
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
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
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

        if self._native_enabled and self._learn_one_native(x, y, w):
            return

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
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
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
        if self._native_model is not None:
            return self._native_model.predict_proba_mapping(
                x,
                self._native_adapter.features,
                self._native_adapter.output_labels,
                self._native_adapter.output_indices,
            )
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    def predict_one(self, x, **kwargs):
        if self._native_model is not None:
            class_index = self._native_model.predict_mapping(
                x,
                self._native_adapter.features,
                self._native_adapter.output_indices,
            )
            return self._native_adapter.classes[class_index]
        return super().predict_one(x, **kwargs)

    @property
    def _multiclass(self):
        return True
