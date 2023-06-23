from __future__ import annotations

from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .nodes.branch import DTBranch
from .nodes.efdtc_nodes import (
    BaseEFDTBranch,
    EFDTLeafMajorityClass,
    EFDTLeafNaiveBayes,
    EFDTLeafNaiveBayesAdaptive,
    EFDTNominalBinaryBranch,
    EFDTNominalMultiwayBranch,
    EFDTNumericBinaryBranch,
    EFDTNumericMultiwayBranch,
)
from .nodes.leaf import HTLeaf
from .splitter import Splitter
from .utils import BranchFactory


class ExtremelyFastDecisionTreeClassifier(HoeffdingTreeClassifier):
    """Extremely Fast Decision Tree classifier.

    Also referred to as Hoeffding AnyTime Tree (HATT) classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    min_samples_reevaluate
        Number of instances a node should observe before reevaluating the best split.
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
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
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
    The Extremely Fast Decision Tree (EFDT) [^1] constructs a tree incrementally. The EFDT seeks to
    select and deploy a split as soon as it is confident the split is useful, and then revisits
    that decision, replacing the split if it subsequently becomes evident that a better split is
    available. The EFDT learns rapidly from a stationary distribution and eventually it learns the
    asymptotic batch tree if the distribution from which the data are drawn is stationary.

    References
    ----------
    [^1]:  C. Manapragada, G. Webb, and M. Salehi. Extremely Fast Decision Tree.
    In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data
    Mining (KDD '18). ACM, New York, NY, USA, 1953-1962.
    DOI: https://doi.org/10.1145/3219819.3220005

    Examples
    --------
    >>> from river.datasets import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.ExtremelyFastDecisionTreeClassifier(
    ...     grace_period=100,
    ...     delta=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode'],
    ...     min_samples_reevaluate=100
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 87.29%
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int | None = None,
        min_samples_reevaluate: int = 20,
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
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.min_samples_reevaluate = min_samples_reevaluate

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau", "min_samples_reevaluate"}

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return EFDTLeafMajorityClass(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return EFDTLeafNaiveBayes(initial_stats, depth, self.splitter)
        else:  # NAIVE BAYES ADAPTIVE (default)
            return EFDTLeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)

    def _branch_selector(self, numerical_feature=True, multiway_split=False) -> type[DTBranch]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return EFDTNumericBinaryBranch
            else:
                return EFDTNumericMultiwayBranch
        else:
            if not multiway_split:
                return EFDTNominalBinaryBranch
            else:
                return EFDTNominalMultiwayBranch

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Incrementally train the model

        Parameters
        ----------
        x
            Instance attributes.
        y
            The label of the instance.
        sample_weight
            The weight of the sample.

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the path from root to the corresponding leaf for
        the instance and sort the instance.
        * Reevaluate the best split for each internal node.
        * Attempt to split the leaf.

        Returns
        -------
        self

        """
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        # Sort instance X into a leaf
        self._sort_to_leaf(x, y, sample_weight)
        # Process all nodes, starting from root to the leaf where the instance x belongs.
        self._process_nodes(x, y, sample_weight, self._root, None, None)

        return self

    def _sort_to_leaf(self, x, y, sample_weight):
        """For a given instance, find the corresponding leaf and update it.

        Private function where leaf learn from instance.

        1. Find the node where instance should be.
        2. If no node have been found, create new learning node.
        3.1 Update the node with the provided instance.

        Parameters
        ----------
        x
            Instance attributes.
        y
            The instance label.
        sample_weight
            The weight of the sample.

        """
        node = self._root
        if isinstance(self._root, DTBranch):
            node = self._root.traverse(x, until_leaf=False)
            # Something went wrong in the way: a missing split feature or emerging category
            # Let's deal with these situations
            if isinstance(node, DTBranch):
                while True:
                    if node.max_branches() == -1 and node.feature in x:
                        # Emerging feature in nominal feature: create a new branch
                        leaf = self._new_leaf(parent=node)
                        node.add_child(x[node.feature], leaf)
                        self._n_active_leaves += 1
                        node = leaf
                    else:
                        # Missing split feature: select the most traversed path to continue the
                        # walk
                        _, node = node.most_common_path()
                        if isinstance(node, DTBranch):
                            node = node.traverse(x, until_leaf=False)
                    if isinstance(node, HTLeaf):
                        break
        node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def _process_nodes(self, x, y, sample_weight, node, parent, branch_index):
        """Process nodes from the root to the leaf where the instance belongs.

        1. If the node is internal:
            1.1 If the number of samples seen since the last reevaluation are greater than
            `min_samples_reevaluate`, reevaluate the best split for the internal node.
        2. If the node is leaf, attempt to split leaf node.

        Parameters
        ----------
        x
            Instance attributes.
        y
            The label of the instance.
        sample_weight
            The weight of the sample.
        node
            The node to process.
        parent
            The node's parent.
        branch_index
            Parent node's branch index.
        """
        if isinstance(node, BaseEFDTBranch):
            # Update split nodes as the tree is traversed
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

            old_weight = node.last_split_reevaluation_at
            new_weight = node.total_weight
            stop_flag = False

            if (new_weight - old_weight) >= self.min_samples_reevaluate:
                # Reevaluate the best split
                stop_flag = self._reevaluate_best_split(
                    node,
                    parent,
                    branch_index,
                    # The attribute observer template that will be replicated to new leaves
                    splitter=self.splitter,
                    # Existing attribute observers that will be leveraged
                    splitters=node.splitters,
                )

            if not stop_flag:
                # Move in depth
                try:
                    child_index = node.branch_no(x)
                    child = node.children[child_index]
                except KeyError:
                    child_index, child = node.most_common_path()
                self._process_nodes(x, y, sample_weight, child, node, child_index)
        elif self._growth_allowed and node.is_active():
            if node.depth >= self.max_depth:  # Max depth reached
                node.deactivate()
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
            else:
                weight_seen = node.total_weight
                weight_diff = weight_seen - node.last_split_attempt_at
                if weight_diff >= self.grace_period:
                    # Attempt to split
                    self._attempt_to_split(
                        node,
                        parent,
                        branch_index,
                        splitter=self.splitter,
                        splitters=node.splitters,
                    )
                    node.last_split_attempt_at = weight_seen

    def _reevaluate_best_split(self, node, parent, branch_index, **kwargs):
        """Reevaluate the best split for a node.

        If the samples seen so far are not from the same class then:
        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the null split candidate is higher than the top split candidate:
            3.1 Kill subtree and replace it with a leaf.
            3.2 Update the tree.
            3.3 Update tree's metrics
        4. If the difference between the top split candidate and the current split is larger than
        the Hoeffding bound:
           4.1 Create a new split node.
           4.2 Update the tree.
           4.3 Update tree's metrics
        5. If the top split candidate is the current split but with different split test:
           5.1 Update the split test of the current split.

        Parameters
        ----------
        node
            The node to reevaluate.
        parent
            The node's parent.
        branch_index
            Parent node's branch index.
        kwargs
            Other parameters passed to the branch node.

        Returns
        -------
            Flag to stop moving in depth.
        """
        stop_flag = False
        if not node.observed_class_distribution_is_pure():
            split_criterion = self._new_split_criterion()
            best_split_suggestions = node.best_split_suggestions(split_criterion, self)
            if len(best_split_suggestions) > 0:
                # Sort the attribute accordingly to their split merit for each attribute
                # (except the null one)
                best_split_suggestions.sort()

                # x_best is the attribute with the highest merit
                x_best = best_split_suggestions[-1]
                id_best = x_best.feature

                # Best split candidate is the null split
                if x_best.feature is None:
                    return True

                # x_current is the current attribute used in this SplitNode
                id_current = node.feature
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = BranchFactory(merit=0)

                # Compute Hoeffding bound
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.delta,
                    node.total_weight,
                )

                if x_null.merit - x_best.merit > hoeffding_bound:
                    # Kill subtree & replace the branch by a leaf
                    best_split = self._kill_subtree(node)

                    # update EFDT
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._root = best_split
                    else:
                        parent.children[branch_index] = best_split

                    n_active = n_inactive = 0
                    for leaf in node.iter_leaves():
                        if leaf.is_active():
                            n_active += 1
                        else:
                            n_inactive += 1

                    self._n_active_leaves += 1
                    self._n_active_leaves -= n_active
                    self._n_inactive_leaves -= n_inactive
                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif (
                    x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound < self.tau
                ) and (id_current != id_best):
                    # Create a new branch
                    branch = self._branch_selector(x_best.numerical_feature, x_best.multiway_split)
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=node)
                        for initial_stats in x_best.children_stats
                    )

                    new_split = x_best.assemble(branch, node.stats, node.depth, *leaves, **kwargs)
                    # Update weights in new_split
                    new_split.last_split_reevaluation_at = node.total_weight

                    n_active = n_inactive = 0
                    for leaf in node.iter_leaves():
                        if leaf.is_active():
                            n_active += 1
                        else:
                            n_inactive += 1

                    self._n_active_leaves -= n_active
                    self._n_inactive_leaves -= n_inactive
                    self._n_active_leaves += len(leaves)

                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._root = new_split
                    else:
                        parent.children[branch_index] = new_split

                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif (
                    x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound < self.tau
                ) and (id_current == id_best):
                    branch = self._branch_selector(x_best.numerical_feature, x_best.multiway_split)
                    # Change the branch but keep the existing children nodes
                    new_split = x_best.assemble(
                        branch, node.stats, node.depth, *tuple(node.children), **kwargs
                    )
                    # Update weights in new_split
                    new_split.last_split_reevaluation_at = node.total_weight

                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._root = new_split
                    else:
                        parent.children[branch_index] = new_split

        return stop_flag

    def _attempt_to_split(self, node, parent, branch_index, **kwargs):
        """Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the difference between the best split candidate and the don't split candidate is
        larger than the Hoeffding bound:
            3.1 Replace the leaf node by a split node.
            3.2 Add a new leaf node on each branch of the new split node.
            3.3 Update tree's metrics

        Parameters
        ----------
        node
            The node to reevaluate.
        parent
            The node's parent.
        branch_index
            Parent node's branch index.
        kwargs
            Other parameters passed to the new branch node.

        """
        if not node.observed_class_distribution_is_pure():  # noqa
            split_criterion = self._new_split_criterion()

            best_split_suggestions = node.best_split_suggestions(split_criterion, self)

            if len(best_split_suggestions) > 0:
                # x_best is the attribute with the highest merit
                best_split_suggestions.sort()
                x_best = best_split_suggestions[-1]

                # Get x_null
                x_null = BranchFactory(merit=0)

                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.delta,
                    node.total_weight,
                )

                if x_best.merit - x_null.merit > hoeffding_bound or hoeffding_bound < self.tau:
                    # Create a new branch
                    branch = self._branch_selector(x_best.numerical_feature, x_best.multiway_split)
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=node)
                        for initial_stats in x_best.children_stats
                    )
                    new_split = x_best.assemble(branch, node.stats, node.depth, *leaves, **kwargs)

                    new_split.last_split_reevaluation_at = node.total_weight

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)

                    if parent is None:
                        # root case: replace the root node by a new split node
                        self._root = new_split
                    else:
                        parent.children[branch_index] = new_split

                    # Manage memory
                    self._enforce_size_limit()

    def _kill_subtree(self, node: BaseEFDTBranch):
        """Kill subtree that starts from node.

        Parameters
        ----------
        node
            The node to reevaluate.
        Returns
        -------
            The new leaf.
        """

        leaf = self._new_leaf()
        leaf.depth = node.depth  # type: ignore
        leaf.stats = node.stats
        leaf.splitters = node.splitters

        return leaf
