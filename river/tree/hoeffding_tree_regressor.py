from copy import deepcopy

from river import base, linear_model

from .hoeffding_tree import HoeffdingTree
from .nodes.branch import HTBranch
from .nodes.htr_nodes import LeafAdaptive, LeafMean, LeafModel
from .nodes.leaf import HTLeaf
from .split_criterion import VarianceReductionSplitCriterion
from .splitter import EBSTSplitter, Splitter


class HoeffdingTreeRegressor(HoeffdingTree, base.Regressor):
    """Hoeffding Tree regressor.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mean' - Target mean</br>
        - 'model' - Uses the model defined in `leaf_model`</br>
        - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
    leaf_model
        The regression model used to provide responses if `leaf_prediction='model'`. If not
        provided an instance of `river.linear_model.LinearRegression` with the default
        hyperparameters is used.
    model_selector_decay
        The exponential decaying factor applied to the learning models' squared errors, that
        are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.EBSTSplitter` is used if `splitter` is `None`.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    binary_split
        If True, only allow binary splits.
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
    The Hoeffding Tree Regressor (HTR) is an adaptation of the incremental tree algorithm of the
    same name for classification. Similarly to its classification counterpart, HTR uses the
    Hoeffding bound to control its split decisions. Differently from the classification algorithm,
    HTR relies on calculating the reduction of variance in the target space to decide among the
    split candidates. The smallest the variance at its leaf nodes, the more homogeneous the
    partitions are. At its leaf nodes, HTR fits either linear models or uses the target
    average as the predictor.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     tree.HoeffdingTreeRegressor(
    ...         grace_period=100,
    ...         leaf_prediction='adaptive',
    ...         model_selector_decay=0.9
    ...     )
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.852902
    """

    _TARGET_MEAN = "mean"
    _MODEL = "model"
    _ADAPTIVE = "adaptive"

    _VALID_LEAF_PREDICTION = [_TARGET_MEAN, _MODEL, _ADAPTIVE]

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "model",
        leaf_model: base.Regressor = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: int = 500,
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

        self._split_criterion: str = "vr"
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.leaf_prediction = leaf_prediction
        self.leaf_model = leaf_model if leaf_model else linear_model.LinearRegression()
        self.model_selector_decay = model_selector_decay
        self.nominal_attributes = nominal_attributes
        self.min_samples_split = min_samples_split

        if splitter is None:
            self.splitter = EBSTSplitter()
        else:
            if splitter.is_target_class:
                raise ValueError(
                    "The chosen splitter cannot be used in regression tasks."
                )
            self.splitter = splitter

    @HoeffdingTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                'Invalid leaf_prediction option "{}", will use default "{}"'.format(
                    leaf_prediction, self._MODEL
                )
            )
            self._leaf_prediction = self._MODEL
        else:
            self._leaf_prediction = leaf_prediction

    @HoeffdingTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != "vr":  # variance reduction
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, "vr"
                )
            )
            self._split_criterion = "vr"
        else:
            self._split_criterion = split_criterion

    def _new_split_criterion(self):
        return VarianceReductionSplitCriterion(min_samples_split=self.min_samples_split)

    def _new_leaf(self, initial_stats=None, parent=None):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = deepcopy(self.leaf_model)
            else:
                try:
                    leaf_model = deepcopy(parent._leaf_model)  # noqa
                except AttributeError:
                    leaf_model = deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return LeafMean(initial_stats, depth, self.splitter)
        elif self.leaf_prediction == self._MODEL:
            return LeafModel(initial_stats, depth, self.splitter, leaf_model)
        else:  # adaptive learning node
            new_adaptive = LeafAdaptive(initial_stats, depth, self.splitter, leaf_model)
            if parent is not None and isinstance(parent, LeafAdaptive):
                new_adaptive._fmse_mean = parent._fmse_mean  # noqa
                new_adaptive._fmse_model = parent._fmse_model  # noqa

            return new_adaptive

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the tree model on sample x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Target value for sample x.
        sample_weight
            The weight of the sample.

        Returns
        -------
        self
        """

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, HTBranch):
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
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = (
                            p_node.branch_no(x)
                            if isinstance(p_node, HTBranch)
                            else None
                        )
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
                    if isinstance(node, HTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, HTLeaf):
                    break
            # Learn from the sample
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_one(self, x):
        """Predict the target value using one of the leaf prediction strategies.

        Parameters
        ----------
        x
            Instance for which we want to predict the target.

        Returns
        -------
        Predicted target value.

        """
        pred = 0.0
        if self._root is not None:
            if isinstance(self._root, HTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            pred = leaf.prediction(x, tree=self)
        return pred

    def _attempt_to_split(
        self, leaf: HTLeaf, parent: HTBranch, parent_branch: int, **kwargs
    ):
        """Attempt to split a node.

        If the target's variance is high at the leaf node, then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the ratio between the merit of the second best split candidate and the merit of the
        best one is smaller than 1 minus the Hoeffding bound (or a tie breaking decision
        takes place), then:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The node to evaluate.
        parent
            The node's parent in the tree.
        parent_branch
            Parent node's branch index.
        kwargs
            Other parameters passed to the new branch.

        """
        split_criterion = self._new_split_criterion()
        best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort()
        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            hoeffding_bound = self._hoeffding_bound(
                split_criterion.range_of_merit(leaf.stats),
                self.split_confidence,
                leaf.total_weight,
            )
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]
            if best_suggestion.merit > 0.0 and (
                second_best_suggestion.merit / best_suggestion.merit
                < 1 - hoeffding_bound
                or hoeffding_bound < self.tie_threshold
            ):
                should_split = True
            if self.remove_poor_attrs:
                poor_attrs = set()
                best_ratio = second_best_suggestion.merit / best_suggestion.merit

                # Add any poor attribute to set
                for suggestion in best_split_suggestions:
                    if (
                        suggestion.feature
                        and suggestion.merit / best_suggestion.merit
                        < best_ratio - 2 * hoeffding_bound
                    ):
                        poor_attrs.add(suggestion.feature)
                for poor_att in poor_attrs:
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
                    for initial_stats in split_decision.children_stats
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
        elif (
            len(best_split_suggestions) >= 2
            and best_split_suggestions[-1].merit > 0
            and best_split_suggestions[-2].merit > 0
        ):
            last_check_ratio = (
                best_split_suggestions[-2].merit / best_split_suggestions[-1].merit
            )
            last_check_vr = best_split_suggestions[-1].merit

            leaf.manage_memory(
                split_criterion, last_check_ratio, last_check_vr, hoeffding_bound
            )
