from operator import attrgetter
import numpy as np

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions

from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._nodes import ActiveLeaf
from ._nodes import LearningNode
from ._nodes import EFDTSplitNode
from ._nodes import EFDTActiveLearningNodeMC
from ._nodes import EFDTInactiveLearningNodeMC
from ._nodes import EFDTActiveLearningNodeNB
from ._nodes import EFDTActiveLearningNodeNBA

import warnings


def HATT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
         min_samples_reevaluate=20, split_criterion='info_gain', split_confidence=0.0000001,
         tie_threshold=0.05, binary_split=False, stop_mem_management=False, leaf_prediction='nba',
         nb_threshold=0, nominal_attributes=None):  # pragma: no cover
    warnings.warn("'HATT' has been renamed to 'ExtremelyFastDecisionTreeClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return ExtremelyFastDecisionTreeClassifier(max_byte_size=max_byte_size,
                                               memory_estimate_period=memory_estimate_period,
                                               grace_period=grace_period,
                                               min_samples_reevaluate=min_samples_reevaluate,
                                               split_criterion=split_criterion,
                                               split_confidence=split_confidence,
                                               tie_threshold=tie_threshold,
                                               binary_split=binary_split,
                                               stop_mem_management=stop_mem_management,
                                               leaf_prediction=leaf_prediction,
                                               nb_threshold=nb_threshold,
                                               nominal_attributes=nominal_attributes)


class ExtremelyFastDecisionTreeClassifier(HoeffdingTreeClassifier):
    """ Extremely Fast Decision Tree classifier.

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
    The Extremely Fast Decision Tree (EFDT) [1]_ constructs a tree incrementally. The EFDT seeks to
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
    >>>
    >>> # Setting up a data stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Setup Extremely Fast Decision Tree classifier
    >>> efdt = ExtremelyFastDecisionTreeClassifier()
    >>>
    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 200
    >>>
    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = efdt.predict(X)
    >>>     if y[0] == y_pred[0]:
    >>>         correct_cnt += 1
    >>>     efdt.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Extremely Fast Decision Tree accuracy: {}'.format(correct_cnt / n_samples))
    """

    # Override _new_learning_node
    def _new_learning_node(self, initial_class_observations=None, is_active=True):
        """ Create a new learning node.

        The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}

        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return EFDTActiveLearningNodeMC(initial_class_observations)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return EFDTActiveLearningNodeNB(initial_class_observations)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return EFDTActiveLearningNodeNBA(initial_class_observations)
        else:
            return EFDTInactiveLearningNodeMC(initial_class_observations)

    # Override _new_split_node
    def _new_split_node(self, split_test, class_observations, attribute_observers):
        """ Create a new split node."""
        return EFDTSplitNode(split_test, class_observations, attribute_observers)

    # =================================================
    # == Extremely Fast Decision Tree implementation ==
    # =================================================

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
                 nominal_attributes=None):

        super().__init__(max_byte_size=max_byte_size,
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
                         nominal_attributes=nominal_attributes)

        self.min_samples_reevaluate = min_samples_reevaluate

    # Override partial_fit
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes
        and their corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length
            of the arrays returned by `predict_proba`
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
        * If the tree is already initialized, find the path from root to the corresponding leaf for
        the instance and sort the instance.

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
            1.2 If new samples seen at last reevaluation are greater than min_samples_reevaluate,
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
        node: EFDTSplitNode or EFDTActiveLearningNode*
            The node to process.
        parent: EFDTSplitNode or EFDTActiveLearningNode*
            The node's parent.
        branch_index: int or None
            Parent node's branch index.

        """

        # skip learning node and update SplitNode because the learning node already learnt from
        # instance.
        if isinstance(node, EFDTSplitNode):
            # update EFDTSplitNode statics & attribute_observers
            node.learn_one(X, y, weight=weight, tree=self)

            # get old statistics
            old_weight = node.get_weight_seen_at_last_split_reevaluation()
            # get new statistics
            new_weight = node.total_weight
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
        elif self._growth_allowed and isinstance(node, ActiveLeaf):
            weight_seen = node.total_weight
            weight_diff = weight_seen - node.last_split_attempt_at
            if weight_diff >= self.grace_period:
                # Attempt to split
                self._attempt_to_split(node, parent, branch_index)
                node.last_split_attempt_at = weight_seen

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
            learning_node.learn_one(X, y, weight=weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_byte_size()

    def _reevaluate_best_split(self, node: EFDTSplitNode, parent, branch_index):
        """ Reevaluate the best split for a node.

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
        node: EFDTSplitNode
            The node to reevaluate.
        parent: EFDTSplitNode
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
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
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
                id_current = node.split_test.get_atts_test_depends_on()[0]
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                #  Compute Hoeffding bound
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.get_range_of_merit(node.stats), self.split_confidence,
                    node.total_weight)

                if x_null.merit - x_best.merit > hoeffding_bound:
                    # Kill subtree & replace the EFDTSplitNode by EFDTActiveLearningNode
                    best_split = self._kill_subtree(node)

                    # update EFDT
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
                    self._enforce_tracker_limit()

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold) and (id_current != id_best):
                    # Create a new branch
                    new_split = self._new_split_node(x_best.split_test, node.stats,
                                                     node.attribute_observers)
                    # Update weights in new_split
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    # Update EFDT
                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_stats_from_split(i))
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
                    self._enforce_tracker_limit()

                elif (x_best.merit - x_current.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold) and (id_current == id_best):
                    node._split_test = x_best.split_test

        return stop_flag

    #  Override _attempt_to_split
    def _attempt_to_split(self, node, parent, branch_index):
        """ Attempt to split a node.

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
        node: EFDTActiveLearningNode
            The node to reevaluate.
        parent: EFDTSplitNode
            The node's parent.
        branch_index: int
            Parent node's branch index.

        """

        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
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

                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.get_range_of_merit(node.stats), self.split_confidence,
                    node.total_weight)

                if (x_best.merit - x_null.merit > hoeffding_bound or hoeffding_bound
                        < self.tie_threshold):
                    # Split
                    new_split = self._new_split_node(x_best.split_test, node.stats,
                                                     node.attribute_observers)

                    # update weights in
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_stats_from_split(i))
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
                    self._enforce_tracker_limit()

    def _kill_subtree(self, node: EFDTSplitNode):
        """ Kill subtree that starts from node.


        Parameters
        ----------
        node: EFDTActiveLearningNode
            The node to reevaluate.
        Returns
        -------
        EFDTActiveLearningNode
            The new leaf.
        """

        leaf = self._new_learning_node()

        leaf.stats = node.stats
        leaf.attribute_observers = node.attribute_observers

        return leaf
