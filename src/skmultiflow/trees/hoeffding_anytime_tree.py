from operator import attrgetter

from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.utils import do_naive_bayes_prediction
from skmultiflow.utils.utils import get_dimensions

Node = HoeffdingTree.Node
SplitNode = HoeffdingTree.SplitNode
ActiveLearningNode = HoeffdingTree.ActiveLearningNode
InactiveLearningNode = HoeffdingTree.InactiveLearningNode
NewNode = HAT.NewNode

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'
error_width_threshold = 300



# Todo : Add memory management
class HoeffdingAnytimeTree(HoeffdingTree):
    """ Hoeffding Anytime Tree or EVDT.

    Parameters
    ----------
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
    binary_split: boolean (default=False)
        If True, only allow binary splits.
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

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1]  C. Manapragada, G. Webb, and M. Salehi. Extremely fast decision tree.
        preprint arXiv:1802.08780, 2018.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    """

    class AnyTimeActiveLearningNode(ActiveLearningNode):
        """ Learning node that supports growth.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """

        def __init__(self, initial_class_observations):
            """ AnyTimeActiveLearningNode class constructor. """
            super().__init__(initial_class_observations)

        # Override
        def get_best_split_suggestions(self, criterion, ht):
            """ Find possible split candidates without taking into account the the null split.

            Parameters
            ----------
            criterion: SplitCriterion
                The splitting criterion to be used.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            list
                Split candidates.

            """
            best_suggestions = []
            pre_split_dist = self._observed_class_distribution

            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, ht.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)

            return best_suggestions

        def get_null_split(self, criterion):
            """ Compute the null split (don't split).

            Parameters
            ----------
            criterion: SplitCriterion
                The splitting criterion to be used.


            Returns
            -------
            list
                Split candidates.

            """

            pre_split_dist = self._observed_class_distribution

            null_split = AttributeSplitSuggestion(None, [{}],
                                                  criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
            return null_split

        def count_nodes(self, split_node_cnt=0):
            """ Calculate the number of split node and leaf starting from this node as a root.

            Returns
            -------
            list[int int]
                [number of split node, number of leaf node].

            """
            return np.array([split_node_cnt, 1])

    class AnyTimeLearningNodeNB(AnyTimeActiveLearningNode):

        def __init__(self, initial_class_observations):
            """ AnyTimeLearningNodeNB class constructor. """
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, ht):
            """ Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self.get_weight_seen() >= ht.nb_threshold:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, ht)

        def disable_attribute(self, att_index):
            """ Disable an attribute observer.

            Disabled in Nodes using Naive Bayes, since poor attributes are used in Naive Bayes calculation.

            Parameters
            ----------
            att_index: int
                Attribute index.

            """
            pass

    class AnyTimeLearningNodeNBAdaptive(AnyTimeLearningNodeNB):

        def __init__(self, initial_class_observations):
            """ AnyTimeLearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """ Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                The instance's weight.
            ht: HoeffdingTree
                The Hoeffding Tree to update.

            """
            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            """ Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    class AnyTimeSplitNode(SplitNode):
        """ Node that splits the data in a Hoeffding Anytime Tree.

        Parameters
        ----------
        split_test: InstanceConditionalTest
            Split test.
        class_observations: dict (class_value, weight) or None
            Class observations
        attribute_observers : dict (attribute id, AttributeClassObserver)
            Attribute Observers
        """

        def __init__(self, split_test, class_observations, attribute_observers):
            """ AnyTimeSplitNode class constructor."""
            super().__init__(split_test, class_observations)
            self._attribute_observers = attribute_observers
            self._weight_seen_at_last_split_reevaluation = 0

        def get_null_split(self, criterion):
            """ Compute the null split (don't split).

            Parameters
            ----------
            criterion: SplitCriterion
                The splitting criterion to be used.

            Returns
            -------
            list
                Split candidates.

            """

            pre_split_dist = self._observed_class_distribution
            null_split = AttributeSplitSuggestion(None, [{}],
                                                  criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
            return null_split

        def get_best_split_suggestions(self, criterion, ht):
            """ Find possible split candidates without taking into account the the null split.

            Parameters
            ----------
            criterion: SplitCriterion
                The splitting criterion to be used.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            list
                Split candidates.

            """

            best_suggestions = []
            pre_split_dist = self._observed_class_distribution

            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, ht.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)

            return best_suggestions

        def find_attribute(self, id_att, split_suggestions):
            """ Find the attribute given the id.

            Parameters
            ----------
            id_att: int.
                Id of attribute to find.
            split_suggestions: list
                Possible split candidates.
            Returns
            -------
            AttributeSplitSuggestion
                Found attribute.
            """

            # return current attribute as AttributeSplitSuggestion
            x_current = None
            for attSplit in split_suggestions:
                selected_id = attSplit.split_test.get_atts_test_depends_on()[0]
                if selected_id == id_att:
                    x_current = attSplit

            return x_current

        def learn_from_instance(self, X, y, weight, ht):
            """ Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """
            # Update attribute_observers
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = NumericAttributeClassObserverGaussian()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            """ Calculate the total weight seen by the node.

            Returns
            -------
            float
                Total weight seen.

            """
            return sum(self._observed_class_distribution.values())

        def get_attribute_observers(self):
            """ Get attribute observers at this node.

            Returns
            -------
            dict (attribute id, AttributeClassObserver)
                Attribute observers of this node.

            """
            return self._attribute_observers

        def get_weight_seen_at_last_split_reevaluation(self):
            """ Get the weight seen at the last split reevaluation.

            Returns
            -------
            float
                Total weight seen at last split reevaluation.

            """
            return self._weight_seen_at_last_split_reevaluation

        def update_weight_seen_at_last_split_reevaluation(self):
            """ Update weight seen at the last split in the reevaluation. """
            self._weight_seen_at_last_split_reevaluation = sum(self._observed_class_distribution.values())

        def count_nodes(self, split_node_cnt=0):
            """ Calculate the number of split node and leaf starting from this node as a root.

            Returns
            -------
            list[int int]
                [number of split node, number of leaf node].

            """

            count = np.array([1, 0])
            # get children
            for branch_idx in range(self.num_children()):
                child = self.get_child(branch_idx)
                if child is not None:
                    count += child.count_nodes(split_node_cnt)
                    split_node_cnt = 0
            return count

    # Override _new_learning_node
    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.AnyTimeActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.AnyTimeLearningNodeNB(initial_class_observations)
        else:
            return self.AnyTimeLearningNodeNBAdaptive(initial_class_observations)

    # Override new_split_node
    def new_split_node(self, split_test, class_observations, attribute_observers):
        """ Create a new split node."""
        return self.AnyTimeSplitNode(split_test, class_observations, attribute_observers)

    # =============================================
    # == Hoeffding Anytime Tree implementation ====
    # =============================================

    def __init__(self,
                 grace_period=200,
                 min_samples_reevaluate=20,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None
                 ):

        super(HoeffdingAnytimeTree, self).__init__(max_byte_size=33554432,
                                                   memory_estimate_period=1000000,
                                                   grace_period=grace_period,
                                                   split_criterion=split_criterion,
                                                   split_confidence=split_confidence,
                                                   tie_threshold=tie_threshold,
                                                   binary_split=binary_split,
                                                   stop_mem_management=False,
                                                   remove_poor_atts=False,
                                                   no_preprune=False,
                                                   leaf_prediction=leaf_prediction,
                                                   nb_threshold=nb_threshold,
                                                   nominal_attributes=nominal_attributes,
                                                   )
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self._tree_root = None
        self.min_samples_reevaluate = min_samples_reevaluate

    # Override partial_fit
    def partial_fit(self, X, y, classes=None, weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes and their
        corresponding targets y.

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

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            if weight is None:
                weight = np.array([1.0])
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(weight)
            if row_cnt != wrow_cnt:
                weight = [weight[0]] * row_cnt
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._train_weight_seen_by_model += weight[i]
                    self._partial_fit(X[i], y[i], weight[i])

    #  Override _partial_fit
    def _partial_fit(self, X, y, weight):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.


        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1

        # Sort instance X into a leaf
        self._sort_instance_into_leaf(X, y, weight)

        # Process all nodes, starting from root to the leaf where the instance X belongs.
        self._process_nodes(X, y, weight, self._tree_root, None, None)

    def _process_nodes(self, X, y, weight, node, parent, branch_index):
        """ Processing nodes from root to leaf where instance belongs.

        Private function.

        From root to leaf node where instance belongs:
            1. If the node is internal:
                1.1 Internal node learn from instance
                1.2 If new samples seen at last reevaluation are more than min_samples_reevaluate, reevaluate the best split for the internal node
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
        if isinstance(node, self.AnyTimeSplitNode):

            # update AnyTimeSplitNode statics & attribute_observers
            node.learn_from_instance(X, y, weight, self)

            # get old statics
            old_weight = node.get_weight_seen_at_last_split_reevaluation()
            # get new statics
            new_weight = node.get_weight_seen()

            if (new_weight - old_weight) >= self.min_samples_reevaluate:
                # Reevaluate the best split
                self._reevaluate_best_split(node, parent, branch_index)

                # Move in depth
            child_index = node.instance_child_index(X)
            if child_index >= 0:
                child = node.get_child(child_index)
                if child is not None:

                    return self._process_nodes(X, y, weight, child, node, child_index)
                else:
                    # Todo : raise error for nominal attribute
                    # This possible if there is a problem with child_index : it returns float in case of nominal attribute
                    pass
        else:
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

        if isinstance(leaf_node, self.LearningNode):
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
                4. If the difference between the top split candidate and the current split is larger than the Hoeffding bound:
                   4.1 Create a new split node.
                   4.2 Update the tree.
                   4.3 Update tree's metrics


                Parameters
                ----------
                node: AnyTimeSplitNode
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

                #  Compute Hoeffding bound
                hoeffding_bound = self.compute_hoeffding_bound(
                    split_criterion.get_range_of_merit(node.get_observed_class_distribution()), self.split_confidence,
                    node.get_weight_seen())

                if x_null.merit - x_best.merit > hoeffding_bound:

                    # Kill subtree & replace the AnyTimeSplitNode by AnyTimeActiveLearningNode

                    best_split = self._kill_subtree(node)

                    # update HATT
                    parent.set_child(branch_index, best_split)

                    deleted_node_cnt = node.count_nodes()

                    self._active_leaf_node_cnt += 1
                    self._active_leaf_node_cnt -= deleted_node_cnt[1]
                    self._decision_node_cnt -= deleted_node_cnt[0]


                elif (x_best.merit - x_current.merit > hoeffding_bound) and (id_current != id_best):

                    # Create a new branch
                    new_split = self.new_split_node(x_best.split_test,
                                                    node.get_observed_class_distribution(),
                                                    node.get_attribute_observers())
                    # Update weights in new_split
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    # Update HAAT
                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += x_best.num_splits()
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

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

                hoeffding_bound = self.compute_hoeffding_bound(
                    split_criterion.get_range_of_merit(node.get_observed_class_distribution()), self.split_confidence,
                    node.get_weight_seen())

                if x_best.merit - x_null.merit > hoeffding_bound:
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

    # Override
    def get_info(self):
        """ Collect information about the Hoeffding Anytime Tree configuration.

        Returns
        -------
        string
            Configuration for the Hoeffding Tree.
        """
        description = type(self).__name__ + ': '
        description += 'grace_period: {} - '.format(self.grace_period)
        description += 'min_samples_reevaluate: {} - '.format(self.min_samples_reevaluate)
        description += 'split_criterion: {} - '.format(self.split_criterion)
        description += 'split_confidence: {} - '.format(self.split_confidence)
        description += 'binary_split: {} - '.format(self.binary_split)
        description += 'leaf_prediction: {} - '.format(self.leaf_prediction)
        description += 'nb_threshold: {} - '.format(self.nb_threshold)
        description += 'nominal_attributes: {} - '.format(self.nominal_attributes)
        return description
