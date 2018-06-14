from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.core.split_criteria.multilabel_info_gain_split_criterion import MutltiLabelInfoGainSplitCriterion
from operator import attrgetter
from skmultiflow.core.utils.utils import *
from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer \
    import GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer \
    import NominalAttributeClassObserver
from skmultiflow.classification.core.split_criteria.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction_multilabel


MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'


class MultilabelHF(HoeffdingTree):

    class MultilabelInactiveNode(HoeffdingTree.InactiveLearningNode):

        def learn_from_instance(self, X, Y, weight, ht):
            self.n_observations += 1

            for l in range(len(Y)):
                if Y[l] == 1:
                    try:
                        self._observed_class_distribution[l] += weight
                    except KeyError:
                        self._observed_class_distribution[l] = weight
                else:
                    try:
                        self._observed_class_distribution[l] += 0
                    except KeyError:
                        self._observed_class_distribution[l] = 0

    class MultilabelLearningNode(HoeffdingTree.ActiveLearningNode):

        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self.n_observations = 0

        def learn_from_instance(self, X, Y, weight, ht):
            self.n_observations += 1
            for l in range(len(Y)):
                if Y[l] == 1:
                    try:
                        self._observed_class_distribution[l] += weight
                    except KeyError:
                        self._observed_class_distribution[l] = weight
                else:
                    try:
                        self._observed_class_distribution[l] += 0
                    except KeyError:
                        self._observed_class_distribution[l] = 0
            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                for l in range(len(Y)):
                    obs.observe_attribute_class(X[i], l, weight if Y[l] == 1 else 0)

    class MultilabelLearningNodeNB(MultilabelLearningNode):

        def __init__(self, initial_class_observations):

            super().__init__(initial_class_observations)

        def get_class_votes(self, X, ht):
            """Get the votes per class for a given instance.

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
                return do_naive_bayes_prediction_multilabel(X, self._observed_class_distribution,
                                                            self._attribute_observers, self.n_observations)
            else:
                return super().get_class_votes(X, ht)

        def disable_attribute(self, att_index):
            """Disable an attribute observer.

            Disabled in Nodes using Naive Bayes, since poor attributes are used in Naive Bayes calculation.

            Parameters
            ----------
            att_index: int
                Attribute index.

            """
            pass

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.MultilabelLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.MultilabelLearningNodeNB(initial_class_observations)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations)

    def _deactivate_learning_node(self, to_deactivate: MultilabelLearningNode, parent: HoeffdingTree.SplitNode, parent_branch: int):
        """Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: ActiveLearningNode
            The node to deactivate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self.MultilabelInactiveNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None):
        super().__init__(max_byte_size,
                         memory_estimate_period,
                         grace_period,
                         split_criterion,
                         split_confidence,
                         tie_threshold,
                         binary_split,
                         stop_mem_management,
                         remove_poor_atts,
                         no_preprune,
                         leaf_prediction,
                         nb_threshold,
                         nominal_attributes)
        self.n_labels = 1

    def _attempt_to_split(self, node: MultilabelLearningNode, parent: HoeffdingTree.SplitNode, parent_idx: int):
        """Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node: ActiveLearningNode
            The node to evaluate.
        parent: SplitNode
            The node's parent.
        parent_idx: int
            Parent node's branch index.

        """
        if not node.observed_class_distribution_is_pure():
            split_criterion = MutltiLabelInfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):    # best_suggestion.merit > 1e-10 and \
                    should_split = True
                if self.remove_poor_atts is not None and self.remove_poor_atts:
                    poor_atts = set()
                    # Scan 1 - add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit > hoeffding_bound:
                                    poor_atts.add(int(split_atts[0]))
                    # Scan 2 - remove good attributes from set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit < hoeffding_bound:
                                    poor_atts.remove(int(split_atts[0]))
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    self._deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self.new_split_node(split_decision.split_test,
                                                    node.get_observed_class_distribution())
                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(split_decision.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += split_decision.num_splits()
                    if parent is None:
                        self._tree_root = new_split
                    else:
                        parent.set_child(parent_idx, new_split)
                # Manage memory
                self.enforce_tracker_limit()

    def get_n_observations_for_instance(self, X):

        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.n_observations
        else:
            return 0

    def predict_proba(self, X):
        """Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        list
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i])
            if votes == {}:
                # Tree is empty, all target_values equal, default to zero
                predictions.append([0]*self.n_labels)
            else:
                y_proba = []
                for l in range(self.n_labels):
                    y_proba.append(votes[l]/self.get_n_observations_for_instance(X[i])
                                   if self.get_n_observations_for_instance(X[i]) != 0 else 0)
                predictions.append(y_proba)
        return predictions

    def predict(self, X):

        pred = []
        Y_proba = self.predict_proba(X)
        for y_proba in Y_proba:
            pred.append([1 if (1 - proba <= proba) else 0 for proba in y_proba])
        return pred

    def partial_fit(self, X, y, classes=None, weight=None):
        """Incrementally trains the model. Train samples (instances) are compossed of X attributes and their
        corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for the instance and update the leaf node
          statistics.
        * If growth is allowed and the number of instances that the leaf has observed between split attempts
          exceed the grace period then attempt to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: Not used.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if y is not None:
            self.n_labels = len(y[0])
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

