from operator import attrgetter

from creme.tree import HoeffdingTreeClassifier

from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._split_criterion import HellingerDistanceCriterion
from ._nodes import ActiveLeaf
from ._nodes import LearningNode
from ._nodes import EFDTSplitNode
from ._nodes import EFDTActiveLearningNodeMC
from ._nodes import EFDTInactiveLearningNodeMC
from ._nodes import EFDTActiveLearningNodeNB
from ._nodes import EFDTActiveLearningNodeNBA


class ExtremelyFastDecisionTreeClassifier(HoeffdingTreeClassifier):
    """Extremely Fast Decision Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    min_samples_reevaluate
        Number of instances a node should observe before reevaluating the best split.
    split_criterion
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Helinger Distance
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    Notes
    -----
    The Extremely Fast Decision Tree (EFDT) [^1] constructs a tree incrementally. The EFDT seeks to
    select and deploy a split as soon as it is confident the split is useful, and then revisits
    that decision, replacing the split if it subsequently becomes evident that a better split is
    available. The EFDT learns rapidly from a stationary distribution and eventually it learns the
    asymptotic batch tree if the distribution from which the data are drawn is stationary.

    References
    ----------
    .. [1]  C. Manapragada, G. Webb, and M. Salehi. Extremely Fast Decision Tree.
       In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data
       Mining (KDD '18). ACM, New York, NY, USA, 1953-1962.
       DOI: https://doi.org/10.1145/3219819.3220005

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier

    >>> # Setting up a data stream
    >>> stream = SEAGenerator(seed=1)

    >>> # Setup Extremely Fast Decision Tree classifier
    >>> efdt = ExtremelyFastDecisionTreeClassifier()

    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 200

    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = efdt.predict(X)
    >>>     if y[0] == y_pred[0]:
    >>>         correct_cnt += 1
    >>>     efdt.partial_fit(X, y)
    >>>     n_samples += 1

    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Extremely Fast Decision Tree accuracy: {}'.format(correct_cnt / n_samples))
    """
    def __init__(self,
                 grace_period: int = 200,
                 min_samples_reevaluate: int = 20,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 **kwargs):

        super().__init__(grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.min_samples_reevaluate = min_samples_reevaluate

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return EFDTActiveLearningNodeMC(initial_stats, depth)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return EFDTActiveLearningNodeNB(initial_stats, depth)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return EFDTActiveLearningNodeNBA(initial_stats, depth)
        else:
            return EFDTInactiveLearningNodeMC(initial_stats, depth)

    def _new_split_node(self, split_test, target_stats, depth, attribute_observers):
        """Create a new split node."""
        return EFDTSplitNode(split_test, target_stats, depth, attribute_observers)

    def learn_one(self, x, y, *, sample_weight=1.):
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
            1.1 Internal node learn from instance
            1.2 If the number of samples seen since the last reevaluation are greater than
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

        # skip learning node and update SplitNode because the learning node already learnt from
        # instance.
        if isinstance(node, EFDTSplitNode):
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

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
        elif self._growth_allowed and isinstance(node, ActiveLeaf):
            if node.depth >= self.max_depth:  # Max depth reached
                self._deactivate_leaf(node, parent, branch_index)
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

        if isinstance(leaf_node, LearningNode):
            learning_node = leaf_node
            learning_node.learn_one(x, y, sample_weight=sample_weight, tree=self)

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

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            if len(best_split_suggestions) > 0:
                # Sort the attribute accordingly to their split merit for each attribute
                # (except the null one)
                best_split_suggestions.sort(key=attrgetter('merit'))

                # x_best is the attribute with the highest merit
                x_best = best_split_suggestions[-1]
                id_best = x_best.split_test.get_atts_test_depends_on()[0]

                # x_current is the current attribute used in this SplitNode
                id_current = node.split_test.get_atts_test_depends_on()[0]
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                #  Compute Hoeffding bound
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.get_range_of_merit(node.stats), self.split_confidence,
                    node.total_weight)

                if x_null.merit - x_best.merit > hoeffding_bound:
                    # Kill subtree & replace the EFDTSplitNode by an EFDTActiveLearningNode
                    best_split = self._kill_subtree(node)

                    # update EFDT
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = best_split
                    else:
                        parent.set_child(branch_index, best_split)

                    deleted_node_cnt = node.count_nodes()

                    self._n_active_leaves += 1
                    self._n_active_leaves -= deleted_node_cnt['leaf_nodes']
                    self._n_decision_nodes -= deleted_node_cnt['decision_nodes']
                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold) and (id_current != id_best):
                    # Create a new branch
                    new_split = self._new_split_node(x_best.split_test, node.stats, node.depth,
                                                     node.attribute_observers)
                    # Update weights in new_split
                    new_split.last_split_reevaluation_at = node.total_weight

                    # Update EFDT
                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_stats_from_split(i))
                        new_split.set_child(i, new_child)

                    deleted_node_cnt = node.count_nodes()

                    self._n_active_leaves -= deleted_node_cnt['leaf_nodes']
                    self._n_decision_nodes -= deleted_node_cnt['decision_nodes']
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

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold) and (id_current == id_best):
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

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)

            if len(best_split_suggestions) > 0:
                # x_best is the attribute with the highest merit
                best_split_suggestions.sort(key=attrgetter('merit'))
                x_best = best_split_suggestions[-1]

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.get_range_of_merit(node.stats), self.split_confidence,
                    node.total_weight)

                if (x_best.merit - x_null.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold):
                    # Split
                    new_split = self._new_split_node(x_best.split_test, node.stats, node.depth,
                                                     node.attribute_observers)

                    new_split.last_split_reevaluation_at = node.total_weight

                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_stats_from_split(i),
                                                            parent=new_split)
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
