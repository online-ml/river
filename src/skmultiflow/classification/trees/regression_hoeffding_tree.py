from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.core.attribute_class_observers.hoeffding_numeric_attribute_class_observer \
    import HoeffdingNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.hoeffding_nominal_class_attribute_observer \
    import HoeffdingNominalAttributeClassObserver
from operator import attrgetter
from skmultiflow.core.utils.utils import *
from skmultiflow.classification.core.split_criteria.variance_reduction_split_criterion \
    import VarianceReductionSplitCriterion


class RegressionHoeffdingTree(HoeffdingTree):

    class InactiveLearningNodeForRegression(HoeffdingTree.InactiveLearningNode):

        def __init__(self, initial_class_observations=None):
            """ LearningNode class constructor. """
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):

            try:
                self._observed_class_distribution[0] += 1
                self._observed_class_distribution[1] += y
                self._observed_class_distribution[0] += y * y
            except KeyError:
                self._observed_class_distribution[0] = 1
                self._observed_class_distribution[0] = y
                self._observed_class_distribution[0] = y * y

    class ActiveLearningNodeForRegression(HoeffdingTree.ActiveLearningNode):

        def __init__(self, initial_class_observations):
            """ ActiveLearningNode class constructor. """
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

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
            try:
                self._observed_class_distribution[0] += 1
                self._observed_class_distribution[1] += y
                self._observed_class_distribution[2] += y * y
            except KeyError:
                self._observed_class_distribution[0] = 1
                self._observed_class_distribution[1] = y
                self._observed_class_distribution[2] = y * y

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = HoeffdingNominalAttributeClassObserver()
                    else:
                        obs = HoeffdingNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            """Calculate the total weight seen by the node.

            Returns
            -------
            float
                Total weight seen.

            """
            if self._observed_class_distribution == {}:
                return 0
            else:
                return self._observed_class_distribution[0]

    def _new_learning_node(self, initial_class_observations=None):
        if initial_class_observations is None:
            initial_class_observations = {}
        return self.ActiveLearningNodeForRegression(initial_class_observations)

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

    def _partial_fit(self, X, y, weight):
        """Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        target_values: Not used.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node
        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._active_leaf_node_cnt += 1
        if isinstance(leaf_node, self.LearningNode):
            learning_node = leaf_node
            learning_node.learn_from_instance(X, y, weight, self)
            if self._growth_allowed and isinstance(learning_node, self.ActiveLearningNodeForRegression):
                active_learning_node = learning_node
                weight_seen = active_learning_node.get_weight_seen()
                weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                if weight_diff >= self.grace_period:
                    self._attempt_to_split(active_learning_node, found_node.parent, found_node.parent_branch)
                    active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()

    def predict(self, X):
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
            votes = self.get_votes_for_instance(X[i]).copy()
            if votes == {}:
                # Tree is empty, all target_values equal, default to zero
                predictions.append([0])
            else:
                number_of_examples_seen = votes[0]
                sum_of_values = votes[1]
                predictions.append(sum_of_values / number_of_examples_seen)

        return predictions

    def predict_proba(self, X):
        pass

    def enforce_tracker_limit(self):
        pass

    def _attempt_to_split(self, node, parent, parent_idx: int):
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
        split_criterion = VarianceReductionSplitCriterion()
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
            if (second_best_suggestion.merit / best_suggestion.merit < 1 - hoeffding_bound
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
