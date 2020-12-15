from operator import attrgetter

from river.tree import HoeffdingTreeClassifier

from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._split_criterion import HellingerDistanceCriterion
from ._nodes import EFDTSplitNode
from ._nodes import EFDTLearningNodeMC
from ._nodes import EFDTLearningNodeNB
from ._nodes import EFDTLearningNodeNBA


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
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
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
    attr_obs
        The Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Parameters can be passed to the AOs (when supported)
        by using `attr_obs_params`. Valid options are:</br>
        - `'bst'`: Binary Search Tree.</br>
        - `'gaussian'`: Gaussian observer. The `n_splits` used to query
         for split candidates can be adjusted (defaults to `10`).</br>
        - `'histogram'`: Histogram-based class frequency estimation.  The number of histogram
        bins (`n_bins` -- defaults to `256`) and the number of split point candidates to
        evaluate (`n_splits` -- defaults to `32`) can be adjusted.</br>
        See 'Notes' for more information about the AOs.
    attr_obs_params
        Parameters passed to the numeric AOs. See `attr_obs` for more information.
    kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

    Notes
    -----
    The Extremely Fast Decision Tree (EFDT) [^1] constructs a tree incrementally. The EFDT seeks to
    select and deploy a split as soon as it is confident the split is useful, and then revisits
    that decision, replacing the split if it subsequently becomes evident that a better split is
    available. The EFDT learns rapidly from a stationary distribution and eventually it learns the
    asymptotic batch tree if the distribution from which the data are drawn is stationary.

    Hoeffding trees rely on Attribute Observer (AO) algorithms to monitor input features
    and perform splits. Nominal features can be easily dealt with, since the partitions
    are well-defined. Numerical features, however, require more sophisticated solutions.
    Currently, three AOs are supported in `river` for classification trees:

    - *Binary Search Tree (BST)*: uses an exhaustive algorithm to find split candidates,
    similarly to batch decision trees. It ends up storing all observations between split
    attempts. This AO is the most costly one in terms of memory and processing
    time; however, it tends to yield the most accurate results when using `leaf_prediction=mc`.
    It cannot be used to calculate the Probability Density Function (PDF) of the monitored
    feature due to its binary tree nature. Hence, leaf prediction strategies other than
    the majority class will end up effectively mimicing the majority class classifier.
    This AO has no parameters.</br>
    - *Gaussian Estimator*: Approximates the numeric feature distribution by using
    a Gaussian distribution per class. The Cumulative Distribution Function (CDF) necessary to
    calculate the entropy (and, consequently, the information gain), the gini index, and
    other split criteria is then calculated using the fit feature's distribution.</br>
    - *Histogram*: approximates the numeric feature distribution using an incrementally
    maintained histogram per class. It represents a compromise between the intensive
    resource usage of BST and the strong assumptions about the feature's distribution
    used in the Gaussian Estimator. Besides that, this AO sits in the middle between the
    previous two in terms of memory usage and running time. Note that the number of
    bins affects the probability density approximation required to use leaves with
    (adaptive) naive bayes models. Hence, Histogram tends to be less accurate than the
    Gaussian estimator when adaptive or naive bayes leaves are used.

    References
    ----------
    [^1]:  C. Manapragada, G. Webb, and M. Salehi. Extremely Fast Decision Tree.
    In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data
    Mining (KDD '18). ACM, New York, NY, USA, 1953-1962.
    DOI: https://doi.org/10.1145/3219819.3220005

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.ExtremelyFastDecisionTreeClassifier(
    ...     grace_period=100,
    ...     split_confidence=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode'],
    ...     min_samples_reevaluate=100
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 89.09%
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        min_samples_reevaluate: int = 20,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        attr_obs: str = "gaussian",
        attr_obs_params: dict = None,
        **kwargs
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            attr_obs=attr_obs,
            attr_obs_params=attr_obs_params,
            **kwargs
        )

        self.min_samples_reevaluate = min_samples_reevaluate

    def _new_learning_node(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return EFDTLearningNodeMC(initial_stats, depth, self.attr_obs, self.attr_obs_params)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return EFDTLearningNodeNB(initial_stats, depth, self.attr_obs, self.attr_obs_params)
        else:  # NAIVE BAYES ADAPTIVE (default)
            return EFDTLearningNodeNBA(initial_stats, depth, self.attr_obs, self.attr_obs_params)

    def _new_split_node(self, split_test, target_stats=None, depth=0, attribute_observers=None):
        """Create a new split node."""
        return EFDTSplitNode(
            split_test=split_test,
            stats=target_stats,
            depth=depth,
            attr_obs=self.attr_obs,
            attr_obs_params=self.attr_obs_params,
            attribute_observers=attribute_observers,
        )

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

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._n_active_leaves = 1

        # Sort instance X into a leaf
        self._sort_instance_into_leaf(x, y, sample_weight)
        # Process all nodes, starting from root to the leaf where the instance x belongs.
        self._process_nodes(x, y, sample_weight, self._tree_root, None, -1)

        return self

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
        if not node.is_leaf():
            old_weight = node.last_split_reevaluation_at
            new_weight = node.total_weight
            stop_flag = False

            if (new_weight - old_weight) >= self.min_samples_reevaluate:
                # Reevaluate the best split
                stop_flag = self._reevaluate_best_split(node, parent, branch_index)

            if not stop_flag:
                # Move in depth
                child_index = node.instance_child_index(x)
                if child_index >= 0:
                    child = node.get_child(child_index)
                    if child is not None:
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
                    self._attempt_to_split(node, parent, branch_index)
                    node.last_split_attempt_at = weight_seen

    def _sort_instance_into_leaf(self, x, y, sample_weight):
        """Sort an instance into a leaf.

        Private function where leaf learn from instance

        1. Find the node where instance should be.
        2. If no node have been found, create new learning node.
        3.1 Update the node with the provided instance

        Parameters
        ----------
        x
            Instance attributes.
        y
            The instance label.
        sample_weight
            The weight of the sample.

        """
        found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
        leaf_node = found_node.node

        if leaf_node is None:
            leaf_node = self._new_learning_node(parent=found_node.parent)
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._n_active_leaves += 1

        leaf_node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def _reevaluate_best_split(self, node, parent, branch_index):
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

        Returns
        -------
            Flag to stop moving in depth.
        """
        stop_flag = False
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            elif self._split_criterion == self._HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()

            best_split_suggestions = node.best_split_suggestions(split_criterion, self)
            if len(best_split_suggestions) > 0:
                # Sort the attribute accordingly to their split merit for each attribute
                # (except the null one)
                best_split_suggestions.sort(key=attrgetter("merit"))

                # x_best is the attribute with the highest merit
                x_best = best_split_suggestions[-1]
                id_best = x_best.split_test.attrs_test_depends_on()[0]

                # x_current is the current attribute used in this SplitNode
                id_current = node.split_test.attrs_test_depends_on()[0]
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = node.null_split(split_criterion)

                # Compute Hoeffding bound
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.split_confidence,
                    node.total_weight,
                )

                if x_null.merit - x_best.merit > hoeffding_bound:
                    # Kill subtree & replace the EFDTSplitNode by an EFDTLearningNode
                    best_split = self._kill_subtree(node)

                    # update EFDT
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = best_split
                    else:
                        parent.set_child(branch_index, best_split)

                    deleted_node_cnt = node.count_nodes()

                    self._n_active_leaves += 1
                    self._n_active_leaves -= deleted_node_cnt["leaf_nodes"]
                    self._n_decision_nodes -= deleted_node_cnt["decision_nodes"]
                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif (
                    x_best.merit - x_current.merit > hoeffding_bound
                    or hoeffding_bound < self.tie_threshold
                ) and (id_current != id_best):
                    # Create a new branch
                    new_split = self._new_split_node(
                        x_best.split_test,
                        node.stats,
                        node.depth,
                        node.attribute_observers,
                    )
                    # Update weights in new_split
                    new_split.last_split_reevaluation_at = node.total_weight

                    # Update EFDT
                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_stats_from_split(i))
                        new_split.set_child(i, new_child)

                    deleted_node_cnt = node.count_nodes()

                    self._n_active_leaves -= deleted_node_cnt["leaf_nodes"]
                    self._n_decision_nodes -= deleted_node_cnt["decision_nodes"]
                    self._n_decision_nodes += 1
                    self._n_active_leaves += x_best.num_splits()

                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif (
                    x_best.merit - x_current.merit > hoeffding_bound
                    or hoeffding_bound < self.tie_threshold
                ) and (id_current == id_best):
                    node._split_test = x_best.split_test

        return stop_flag

    def _attempt_to_split(self, node, parent, branch_index):
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

        """
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            elif self._split_criterion == self._HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()

            best_split_suggestions = node.best_split_suggestions(split_criterion, self)

            if len(best_split_suggestions) > 0:
                # x_best is the attribute with the highest merit
                best_split_suggestions.sort(key=attrgetter("merit"))
                x_best = best_split_suggestions[-1]

                # Get x_null
                x_null = node.null_split(split_criterion)

                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.split_confidence,
                    node.total_weight,
                )

                if (
                    x_best.merit - x_null.merit > hoeffding_bound
                    or hoeffding_bound < self.tie_threshold
                ):
                    # Split
                    new_split = self._new_split_node(
                        x_best.split_test,
                        node.stats,
                        node.depth,
                        node.attribute_observers,
                    )

                    new_split.last_split_reevaluation_at = node.total_weight

                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(
                            x_best.resulting_stats_from_split(i), parent=new_split
                        )
                        new_split.set_child(i, new_child)
                    self._n_active_leaves -= 1
                    self._n_decision_nodes += 1
                    self._n_active_leaves += x_best.num_splits()

                    if parent is None:
                        # root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

                    # Manage memory
                    self._enforce_size_limit()

    def _kill_subtree(self, node: EFDTSplitNode):
        """Kill subtree that starts from node.

        Parameters
        ----------
        node
            The node to reevaluate.
        Returns
        -------
            The new leaf.
        """

        leaf = self._new_learning_node()
        leaf.depth = node.depth

        leaf.stats = node.stats
        leaf.attribute_observers = node.attribute_observers

        return leaf
