from operator import attrgetter
import numpy as np
from skmultiflow.trees.split_criterion import GiniSplitCriterion
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.trees.split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.nodes import LearningNode
from skmultiflow.trees.nodes import AnyTimeSplitNode
from skmultiflow.trees.nodes import AnyTimeActiveLearningNode
from skmultiflow.trees.nodes import AnyTimeInactiveLearningNode
from skmultiflow.trees.nodes import AnyTimeLearningNodeNB
from skmultiflow.trees.nodes import AnyTimeLearningNodeNBAdaptive
from skmultiflow.utils import get_dimensions, calculate_object_size


GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'


class HATT(HoeffdingTree):
    """ Hoeffding Anytime Tree or Extremely Fast Decision Tree.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    min_samples_reevaluate: int (default=20)
        Number of instances a node should observe before reevaluate the best split.
    split_criterion: string (default='info_gain')
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.
    binary_split: boolean (default=False)
        If True, only allow binary splits.
    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    Hoeffding Anytime Tree or Extremely Fast Decision Tree (EFDT) [1]_ constructs a tree incrementally. HATT seeks
    to select and deploy a split as soon as it is confident the split is useful, and then revisits that decision,
    replacing the split if it subsequently becomes evident that a better split is available. HATT learns rapidly from a
    stationary distribution and eventually it learns the asymptotic batch tree if the distribution from which the data
    are drawn is stationary.

    References
    ----------
    .. [1]  C. Manapragada, G. Webb, and M. Salehi. Extremely fast decision tree.
       preprint arXiv:1802.08780, 2018.

    """
    # Override _new_learning_node
    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return AnyTimeActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return AnyTimeLearningNodeNB(initial_class_observations)
        else:  # NAIVE BAYES ADAPTIVE (default)
            return AnyTimeLearningNodeNBAdaptive(initial_class_observations)

    # Override new_split_node
    def new_split_node(self, split_test, class_observations, attribute_observers):
        """ Create a new split node."""
        return AnyTimeSplitNode(split_test, class_observations, attribute_observers)

    # =============================================
    # == Hoeffding Anytime Tree implementation ====
    # =============================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 min_samples_reevaluate=20,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None
                 ):

        super(HATT, self).__init__(max_byte_size=max_byte_size,
                                   memory_estimate_period=memory_estimate_period,
                                   grace_period=grace_period,
                                   split_criterion=split_criterion,
                                   split_confidence=split_confidence,
                                   tie_threshold=tie_threshold,
                                   binary_split=binary_split,
                                   stop_mem_management=stop_mem_management,
                                   remove_poor_atts=False,
                                   no_preprune=False,
                                   leaf_prediction=leaf_prediction,
                                   nb_threshold=nb_threshold,
                                   nominal_attributes=nominal_attributes,
                                   )

        self.min_samples_reevaluate = min_samples_reevaluate

    # Override partial_fit
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes and their
        corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        Notes
        -----
        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the path from root to the corresponding leaf for the instance and
          sort the instance.

          * Reevaluate the best split for each internal node.
          * Attempt to split the leaf.
        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            if sample_weight is None:
                sample_weight = np.array([1.0])
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(sample_weight)
            if row_cnt != wrow_cnt:
                sample_weight = [sample_weight[0]] * row_cnt
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self._partial_fit(X[i], y[i], sample_weight[i])

    #  Override _partial_fit
    def _partial_fit(self, X, y, sample_weight):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: int
            Class label for sample X.
        sample_weight: float
            Sample weight.

        """

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1

        # Sort instance X into a leaf
        self._sort_instance_into_leaf(X, y, sample_weight)

        # Process all nodes, starting from root to the leaf where the instance X belongs.
        self._process_nodes(X, y, sample_weight, self._tree_root, None, None)

    def _process_nodes(self, X, y, weight, node, parent, branch_index):
        """ Processing nodes from root to leaf where instance belongs.

        Private function.

        From root to leaf node where instance belongs:
        1. If the node is internal:
            1.1 Internal node learn from instance
            1.2 If new samples seen at last reevaluation are more than min_samples_reevaluate,
                reevaluate the best split for the internal node
        2. If the node is leaf, attempt to split leaf node.


        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        node: AnyTimeSplitNode or AnyTimeActiveLearningNode
            The node to process.
        parent: AnyTimeSplitNode or AnyTimeActiveLearningNode
            The node's parent.
        branch_index: int or None
            Parent node's branch index.

        """

        # skip learning node and update SplitNode because the learning node already learnt from instance.
        if isinstance(node, AnyTimeSplitNode):

            # update AnyTimeSplitNode statics & attribute_observers
            node.learn_from_instance(X, y, weight, self)

            # get old statics
            old_weight = node.get_weight_seen_at_last_split_reevaluation()
            # get new statics
            new_weight = node.get_weight_seen()

            stop_flag = False

            if (new_weight - old_weight) >= self.min_samples_reevaluate:
                # Reevaluate the best split
                stop_flag = self._reevaluate_best_split(node, parent, branch_index)

            if not stop_flag:
                # Move in depth
                child_index = node.instance_child_index(X)
                if child_index >= 0:
                    child = node.get_child(child_index)
                    if child is not None:

                        self._process_nodes(X, y, weight, child, node, child_index)
                    else:
                        # Todo : raise error for nominal attribute
                        # This possible if there is a problem with child_index, it returns float in case of
                        # nominal attribute
                        pass
        elif self._growth_allowed and isinstance(node, AnyTimeActiveLearningNode):
            weight_seen = node.get_weight_seen()
            weight_diff = weight_seen - node.get_weight_seen_at_last_split_evaluation()
            if weight_diff >= self.grace_period:
                # Attempt to split
                self._attempt_to_split(node, parent, branch_index)
                node.set_weight_seen_at_last_split_evaluation(weight_seen)

    def _sort_instance_into_leaf(self, X, y, weight):
        """ Sort an instance into a leaf.

        Private function where leaf learn from instance:

        1. Find the node where instance should be.
        2. If no node have been found, create new learning node.
        3.1 Update the node with the provided instance

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node

        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._active_leaf_node_cnt += 1

        if isinstance(leaf_node, LearningNode):
            learning_node = leaf_node
            learning_node.learn_from_instance(X, y, weight, self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()

    def _reevaluate_best_split(self, node: AnyTimeSplitNode, parent, branch_index):
        """ Reevaluate the best split for a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the don't split candidate is higher than the top split candidate:
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
        node: AnyTimeSplitNode
            The node to reevaluate.
        parent: AnyTimeSplitNode
            The node's parent.
        branch_index: int
            Parent node's branch index.
        Returns
        -------
        boolean
            flag to stop moving in depth.
        """

        stop_flag = False
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            if len(best_split_suggestions) > 0:

                # Compute Gini (or Information Gain) for each attribute (except the null one)
                best_split_suggestions.sort(key=attrgetter('merit'))
                # x_best is the attribute with the highest G_int

                x_best = best_split_suggestions[-1]
                id_best = x_best.split_test.get_atts_test_depends_on()[0]

                # x_current is the current attribute used in this SplitNode
                id_current = node.get_split_test().get_atts_test_depends_on()[0]
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                # Force x_null merit to get 0 instead of -infinity
                if x_null.merit == -np.inf:
                    x_null.merit = 0.0

                #  Compute Hoeffding bound
                hoeffding_bound = self.compute_hoeffding_bound(
                    split_criterion.get_range_of_merit(node.get_observed_class_distribution()), self.split_confidence,
                    node.get_weight_seen())

                if x_null.merit - x_best.merit > hoeffding_bound:

                    # Kill subtree & replace the AnyTimeSplitNode by AnyTimeActiveLearningNode

                    best_split = self._kill_subtree(node)

                    # update HATT
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = best_split
                    else:
                        parent.set_child(branch_index, best_split)

                    deleted_node_cnt = node.count_nodes()

                    self._active_leaf_node_cnt += 1
                    self._active_leaf_node_cnt -= deleted_node_cnt[1]
                    self._decision_node_cnt -= deleted_node_cnt[0]
                    stop_flag = True

                    # Manage memory
                    self.enforce_tracker_limit()

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound < self.tie_threshold) and (
                        id_current != id_best):

                    # Create a new branch
                    new_split = self.new_split_node(x_best.split_test,
                                                    node.get_observed_class_distribution(),
                                                    node.get_attribute_observers())
                    # Update weights in new_split
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    # Update HATT
                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)

                    deleted_node_cnt = node.count_nodes()

                    self._active_leaf_node_cnt -= deleted_node_cnt[1]
                    self._decision_node_cnt -= deleted_node_cnt[0]
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += x_best.num_splits()

                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

                    stop_flag = True

                    # Manage memory
                    self.enforce_tracker_limit()

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound < self.tie_threshold) and (
                        id_current == id_best):
                    node._split_test = x_best.split_test

        return stop_flag

    #  Override _attempt_to_split
    def _attempt_to_split(self, node, parent, branch_index):
        """ Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the difference between the best split candidate and the don't split candidate is larger than
        the Hoeffding bound:
            3.1 Replace the leaf node by a split node.
            3.2 Add a new leaf node on each branch of the new split node.
            3.3 Update tree's metrics

        Parameters
        ----------
        node: AnyTimeActiveLearningNode
            The node to reevaluate.
        parent: AnyTimeSplitNode
            The node's parent.
        branch_index: int
            Parent node's branch index.

        """

        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)

            if len(best_split_suggestions) > 0:

                # x_best is the attribute with the highest G_int
                best_split_suggestions.sort(key=attrgetter('merit'))
                x_best = best_split_suggestions[-1]

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                # Force x_null merit to get 0 instead of -infinity
                if x_null.merit == -np.inf:
                    x_null.merit = 0.0

                hoeffding_bound = self.compute_hoeffding_bound(
                    split_criterion.get_range_of_merit(node.get_observed_class_distribution()), self.split_confidence,
                    node.get_weight_seen())

                if x_best.merit - x_null.merit > hoeffding_bound or hoeffding_bound < self.tie_threshold:

                    # Split
                    new_split = self.new_split_node(x_best.split_test,
                                                    node.get_observed_class_distribution(),
                                                    node.get_attribute_observers())

                    # update weights in
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += x_best.num_splits()

                    if parent is None:
                        # root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

                    # Manage memory
                    self.enforce_tracker_limit()

    def _kill_subtree(self, node: AnyTimeSplitNode):
        """ Kill subtree that starts from node.


        Parameters
        ----------
        node: AnyTimeActiveLearningNode
            The node to reevaluate.
        Returns
        -------
        AnyTimeActiveLearningNode
            The new leaf.
        """

        leaf = self._new_learning_node()

        leaf.set_observed_class_distribution(node.get_observed_class_distribution())
        leaf.set_attribute_observers(node.get_attribute_observers())

        return leaf

    # Override enforce_tracker_limit
    def enforce_tracker_limit(self):
        """ Track the size of the tree and disable/enable nodes if required."""
        byte_size = (self._active_leaf_byte_size_estimate
                     + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate) \
                    * self._byte_size_estimate_overhead_fraction

        if self._inactive_leaf_node_cnt > 0 or byte_size > self.max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        learning_nodes = self._find_learning_nodes()
        learning_nodes.sort(key=lambda n: n.node.calculate_promise())
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if ((max_active * self._active_leaf_byte_size_estimate + (len(learning_nodes) - max_active)
                 * self._inactive_leaf_byte_size_estimate) * self._byte_size_estimate_overhead_fraction) \
                    > self.max_byte_size:
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if isinstance(learning_nodes[i].node, AnyTimeActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)
        for i in range(cutoff, len(learning_nodes)):
            if isinstance(learning_nodes[i].node, AnyTimeInactiveLearningNode):
                self._activate_learning_node(learning_nodes[i].node,
                                             learning_nodes[i].parent,
                                             learning_nodes[i].parent_branch)

    # Override _estimate_model_byte_size
    def estimate_model_byte_size(self):
        """ Calculate the size of the model and trigger tracker function if the actual model size exceeds the max size
        in the configuration."""
        learning_nodes = self._find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if isinstance(found_node.node, AnyTimeActiveLearningNode):
                total_active_size += calculate_object_size(found_node.node)
            else:
                total_inactive_size += calculate_object_size(found_node.node)
        if total_active_size > 0:
            self._active_leaf_byte_size_estimate = total_active_size / self._active_leaf_node_cnt
        if total_inactive_size > 0:
            self._inactive_leaf_byte_size_estimate = total_inactive_size / self._inactive_leaf_node_cnt
        actual_model_size = calculate_object_size(self)
        estimated_model_size = (self._active_leaf_node_cnt * self._active_leaf_byte_size_estimate
                                + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate)
        self._byte_size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self.max_byte_size:
            self.enforce_tracker_limit()

    #  Override deactivate_all_leaves
    def deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for i in range(len(learning_nodes)):
            if isinstance(learning_nodes[i], AnyTimeActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)

    # Override _deactivate_learning_node
    def _deactivate_learning_node(self, to_deactivate: AnyTimeActiveLearningNode, parent: AnyTimeSplitNode,
                                  parent_branch: int):
        """ Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: AnyTimeActiveLearningNode
            The node to deactivate.
        parent:  AnyTimeSplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = AnyTimeInactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    # Override _activate_learning_node
    def _activate_learning_node(self, to_activate: AnyTimeInactiveLearningNode, parent: AnyTimeSplitNode,
                                parent_branch: int):
        """ Activate a learning node.

        Parameters
        ----------
        to_activate: AnyTimeInactiveLearningNode
            The node to activate.
        parent: AnyTimeSplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(to_activate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    # Override measure_byte_size
    def measure_byte_size(self):
        """ Calculate the size of the tree.

        Returns
        -------
        int
            Size of the tree in bytes.

        """
        return calculate_object_size(self)

    # Override rest
    def reset(self):
        """ Reset the Hoeffding Anytime Tree to default values."""
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

    def get_model_description(self):
        """ Walk the tree and return its structure in a buffer.

        Returns
        -------
        string
            The description of the model.

        """
        if self._tree_root is not None:
            buffer = ['']
            description = ''
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description
