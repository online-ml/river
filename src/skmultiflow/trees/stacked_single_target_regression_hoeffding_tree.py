import numpy as np
from skmultiflow.trees.regression_hoeffding_tree import \
    RegressionHoeffdingTree
from skmultiflow.trees.multi_target_regression_hoeffding_tree import \
    MultiTargetRegressionHoeffdingTree
from skmultiflow.trees.hoeffding_numeric_attribute_class_observer \
     import HoeffdingNumericAttributeClassObserver
from skmultiflow.trees.hoeffding_nominal_class_attribute_observer \
     import HoeffdingNominalAttributeClassObserver
from operator import attrgetter
from skmultiflow.utils.utils import get_dimensions
from skmultiflow.trees.intra_cluster_variance_reduction_split_criterion \
     import IntraClusterVarianceReductionSplitCriterion
import logging

TARGET_MEAN = 'tm'
PERCEPTRON = 'perceptron'
ADAPTIVE = 'adaptive'

# logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class StackedSingleTargetRegressionHoeffdingTree(
        MultiTargetRegressionHoeffdingTree):
    """
    Stacked Single Target Multi-target Regression Hoeffding tree.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.
    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.
    binary_split: boolean (default=False)
        If True, only allow binary splits.
    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.
    no_preprune: boolean (default=False)
        If True, disable pre-pruning.
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'tm' - Target mean
        | 'perceptron' - Perceptron
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes
        are numerical.
    learning_ratio_perceptron: float
        The learning rate of the perceptron.
    learning_ratio_decay: float
        Decay multiplier for the learning rate of the perceptron
    learning_ratio_const: Bool
        If False the learning ratio will decay with the number of examples seen
    """

    class LearningNodePerceptron(MultiTargetRegressionHoeffdingTree.
                                 LearningNodePerceptron):

        def __init__(self, initial_class_observations, perceptron_weight=None):
            """
            LearningNodePerceptron class constructor
            Parameters
            ----------
            initial_class_observations
            perceptron_weight
            """
            super().__init__(initial_class_observations)

            if perceptron_weight is None:
                self.perceptron_weight = {}
            else:
                self.perceptron_weight = perceptron_weight

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: numpy.ndarray of length equal to the number of targets.
                Instance targets.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """

            if self.perceptron_weight == {}:
                # Creates matrix of perceptron random weights
                _, rows = get_dimensions(y)
                _, cols = get_dimensions(X)

                self.perceptron_weight[0] = np.random.uniform(-1.0, 1.0,
                                                              (rows, cols + 1))
                # Cascade Stacking
                self.perceptron_weight[1] = np.random.uniform(-1.0, 1.0,
                                                              (rows, rows + 1))
                self.normalize_perceptron_weights()

            try:
                self._observed_class_distribution[0] += weight
            except KeyError:
                self._observed_class_distribution[0] = weight

            if ht.learning_ratio_const:
                learning_ratio = ht.learning_ratio_perceptron
            else:
                learning_ratio = ht.learning_ratio_perceptron / \
                                 (1 + self._observed_class_distribution[0] *
                                  ht.learning_ratio_decay)

            try:
                self._observed_class_distribution[1] += weight * y
                self._observed_class_distribution[2] += weight * (y ** 2)
            except KeyError:
                self._observed_class_distribution[1] = weight * y
                self._observed_class_distribution[2] = weight * (y ** 2)

            for i in range(int(weight)):
                self.update_weights(X, y, learning_ratio, ht)

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    # Creates targets observers, if not already defined
                    if i in ht.nominal_attributes:
                        obs = HoeffdingNominalAttributeClassObserver()
                    else:
                        obs = HoeffdingNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], y, weight)

        def update_weights(self, X, y, learning_ratio, ht):
            """
            Update the perceptron weights
            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: numpy.ndarray of length equal to the number of targets.
                Targets values.
            learning_ratio: float
                perceptron learning ratio
            ht: HoeffdingTree
                Hoeffding Tree to update.
            """
            normalized_sample = ht.normalize_sample(X)
            normalized_base_pred = self._predict_base(normalized_sample)

            _, n_features = get_dimensions(X)
            _, n_targets = get_dimensions(y)

            normalized_target_value = ht.normalized_target_value(y)

            self.perceptron_weight[0] += learning_ratio * \
                (normalized_target_value - normalized_base_pred).\
                reshape((n_targets, 1)) @ \
                normalized_sample.reshape((1, n_features + 1))

            # Add bias term
            normalized_base_pred = np.append(normalized_base_pred, 1.0)
            normalized_meta_pred = self._predict_meta(normalized_base_pred)

            self.perceptron_weight[1] += learning_ratio * \
                (normalized_target_value - normalized_meta_pred).\
                reshape((n_targets, 1)) @ \
                normalized_base_pred.reshape((1, n_targets + 1))

            self.normalize_perceptron_weights()

        # Normalize both levels
        def normalize_perceptron_weights(self):
            n_targets = self.perceptron_weight[0].shape[0]
            # Normalize perceptron weights
            for i in range(n_targets):
                sum_w_0 = np.sum(np.absolute(self.perceptron_weight[0][i, :]))
                self.perceptron_weight[0][i, :] /= sum_w_0
                sum_w_1 = np.sum(np.absolute(self.perceptron_weight[1][i, :]))
                self.perceptron_weight[1][i, :] /= sum_w_1

        def _predict_base(self, X):
            return self.perceptron_weight[0] @ X

        def _predict_meta(self, X):
            return self.perceptron_weight[1] @ X

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

    class LearningNodeAdaptive(LearningNodePerceptron):

        def __init__(self, initial_class_observations, perceptron_weight=None):
            """
            LearningNodeAdaptive class constructor
            Parameters
            ----------
            initial_class_observations
            perceptron_weight
            """
            super().__init__(initial_class_observations, perceptron_weight)

            # Faded adaptive errors
            self.fMAE_M = 0.0
            self.fMAE_P = 0.0
            # Stacked Perceptron
            self.fMAE_SP = 0.0

        def update_weights(self, X, y, learning_ratio, ht):
            """
            Update the perceptron weights
            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: numpy.ndarray of length equal to the number of targets.
                Targets values.
            learning_ratio: float
                perceptron learning ratio
            ht: HoeffdingTree
                Hoeffding Tree to update.
            """
            normalized_sample = ht.normalize_sample(X)
            normalized_base_pred = self._predict_base(normalized_sample)

            _, n_features = get_dimensions(X)
            _, n_targets = get_dimensions(y)

            normalized_target_value = ht.normalized_target_value(y)

            self.perceptron_weight[0] += learning_ratio * \
                (normalized_target_value - normalized_base_pred).\
                reshape((n_targets, 1)) @ \
                normalized_sample.reshape((1, n_features + 1))

            # Add bias term
            normalized_base_pred = np.append(normalized_base_pred, 1.0)
            normalized_meta_pred = self._predict_meta(normalized_base_pred)

            self.perceptron_weight[1] += learning_ratio * \
                (normalized_target_value - normalized_meta_pred).\
                reshape((n_targets, 1)) @ \
                normalized_base_pred.reshape((1, n_targets + 1))

            self.normalize_perceptron_weights()

            # Update faded errors for the predictors
            # The considered errors are normalized, since they are based on
            # mean centered and sd scaled values
            self.fMAE_M = 0.95 * self.fMAE_M + np.absolute(
                normalized_target_value - ht.
                normalized_target_value(self._observed_class_distribution[1] /
                                        self._observed_class_distribution[0])
            )

            # Ignore added bias term in the comparison
            self.fMAE_P = 0.95 * self.fMAE_P + np.absolute(
                normalized_target_value - normalized_base_pred[:-1]
            )

            self.fMAE_SP = 0.95 * self.fMAE_SP + np.absolute(
                normalized_target_value - normalized_meta_pred
            )

    class InactiveLearningNodePerceptron(MultiTargetRegressionHoeffdingTree.
                                         InactiveLearningNodePerceptron):

        def __init__(self, initial_class_observations, perceptron_weight=None):
            super().__init__(initial_class_observations)
            if perceptron_weight is None:
                self.perceptron_weight = {}
            else:
                self.perceptron_weight = perceptron_weight

        def learn_from_instance(self, X, y, weight, ht):

            if self.perceptron_weight == {}:
                # Creates matrix of perceptron random weights
                _, rows = get_dimensions(y)
                _, cols = get_dimensions(X)

                self.perceptron_weight[0] = np.random.uniform(-1.0, 1.0,
                                                              (rows, cols + 1))
                # Cascade Stacking
                self.perceptron_weight[1] = np.random.uniform(-1.0, 1.0,
                                                              (rows, rows + 1))
                self.normalize_perceptron_weights()

            try:
                self._observed_class_distribution[0] += weight
            except KeyError:
                self._observed_class_distribution[0] = weight

            if ht.learning_ratio_const:
                learning_ratio = ht.learning_ratio_perceptron
            else:
                learning_ratio = ht.learning_ratio_perceptron / \
                                (1 + self._observed_class_distribution[0] *
                                 ht.learning_ratio_decay)

            try:
                self._observed_class_distribution[1] += weight * y
                self._observed_class_distribution[2] += weight * (y ** 2)
            except KeyError:
                self._observed_class_distribution[1] = weight * y
                self._observed_class_distribution[2] = weight * (y ** 2)

            for i in range(int(weight)):
                self.update_weights(X, y, learning_ratio, ht)

        def update_weights(self, X, y, learning_ratio, ht):
            """
            Update the perceptron weights
            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: numpy.ndarray of length equal to the number of targets.
                Targets values.
            learning_ratio: float
                perceptron learning ratio
            ht: HoeffdingTree
                Hoeffding Tree to update.
            """
            normalized_sample = ht.normalize_sample(X)
            normalized_base_pred = self._predict_base(normalized_sample)

            _, n_features = get_dimensions(X)
            _, n_targets = get_dimensions(y)

            normalized_target_value = ht.normalized_target_value(y)

            self.perceptron_weight[0] += learning_ratio * \
                (normalized_target_value - normalized_base_pred).\
                reshape((n_targets, 1)) @ \
                normalized_sample.reshape((1, n_features + 1))

            # Add bias term
            normalized_base_pred = np.append(normalized_base_pred, 1.0)
            normalized_meta_pred = self._predict_meta(normalized_base_pred)

            self.perceptron_weight[1] += learning_ratio * \
                (normalized_target_value - normalized_meta_pred).\
                reshape((n_targets, 1)) @ \
                normalized_base_pred.reshape((1, n_targets + 1))

            self.normalize_perceptron_weights()

        # Normalize both levels
        def normalize_perceptron_weights(self):
            n_targets = self.perceptron_weight[0].shape[0]
            # Normalize perceptron weights
            for i in range(n_targets):
                sum_w_0 = np.sum(np.absolute(self.perceptron_weight[0][i, :]))
                self.perceptron_weight[0][i, :] /= sum_w_0
                sum_w_1 = np.sum(np.absolute(self.perceptron_weight[1][i, :]))
                self.perceptron_weight[1][i, :] /= sum_w_1

        def _predict_base(self, X):
            return self.perceptron_weight[0] @ X

        def _predict_meta(self, X):
            return self.perceptron_weight[1] @ X

    class InactiveLearningNodeAdaptive(InactiveLearningNodePerceptron):

        def __init__(self, initial_class_observations, perceptron_weight=None):
            """
            InactiveLearningNodeAdaptive class constructor
            Parameters
            ----------
            initial_class_observations
            perceptron_weight
            """
            super().__init__(initial_class_observations, perceptron_weight)

            # Faded adaptive errors
            self.fMAE_M = 0.0
            self.fMAE_P = 0.0
            # Stacked Perceptron
            self.fMAE_SP = 0.0

        def update_weights(self, X, y, learning_ratio, ht):
            """
            Update the perceptron weights
            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: numpy.ndarray of length equal to the number of targets.
                Targets values.
            learning_ratio: float
                perceptron learning ratio
            ht: HoeffdingTree
                Hoeffding Tree to update.
            """
            normalized_sample = ht.normalize_sample(X)
            normalized_base_pred = self._predict_base(normalized_sample)

            _, n_features = get_dimensions(X)
            _, n_targets = get_dimensions(y)

            normalized_target_value = ht.normalized_target_value(y)

            self.perceptron_weight[0] += learning_ratio * \
                (normalized_target_value - normalized_base_pred).\
                reshape((n_targets, 1)) @ \
                normalized_sample.reshape((1, n_features + 1))

            # Add bias term
            normalized_base_pred = np.append(normalized_base_pred, 1.0)
            normalized_meta_pred = self._predict_meta(normalized_base_pred)

            self.perceptron_weight[1] += learning_ratio * \
                (normalized_target_value - normalized_meta_pred).\
                reshape((n_targets, 1)) @ \
                normalized_base_pred.reshape((1, n_targets + 1))

            self.normalize_perceptron_weights()

            # Update faded errors for the predictors
            # The considered errors are normalized, since they are based on
            # mean centered and sd scaled values
            self.fMAE_M = 0.95 * self.fMAE_M + np.absolute(
                normalized_target_value - ht.
                normalized_target_value(self._observed_class_distribution[1] /
                                        self._observed_class_distribution[0])
            )

            # Ignore added bias term in the comparison
            self.fMAE_P = 0.95 * self.fMAE_P + np.absolute(
                normalized_target_value - normalized_base_pred[:-1]
            )

            self.fMAE_SP = 0.95 * self.fMAE_SP + np.absolute(
                normalized_target_value - normalized_meta_pred
            )

    # ===========================================
    # == Hoeffding Regression Tree implementation ===
    # ===========================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 leaf_prediction="perceptron",
                 no_preprune=False,
                 nb_threshold=0,
                 nominal_attributes=None,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True):

        self.split_criterion = 'intra cluster variance reduction'
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const
        self.examples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0
        self.sum_of_attribute_values = 0.0
        self.sum_of_attribute_squares = 0.0
        self.leaf_prediction = leaf_prediction

        # To add the n_targets property once
        self._n_targets_set = False

    def _get_predictors_faded_error(self, X):
        """ Get the faded error of the leaf corresponding to the pased instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (predictor, fmae)

        """
        fmaes = {}
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            fmaes['mean'] = leaf_node.fMAE_M
            fmaes['perceptron'] = leaf_node.fMAE_P
            fmaes['stacked_perceptron'] = leaf_node.fMAE_SP

        return fmaes

    def predict(self, X):
        """Predicts the target value using mean class or the perceptron.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        list
            Predicted target values.

        """
        r, _ = get_dimensions(X)

        predictions = np.zeros((r, self._n_targets), dtype=np.float64)
        for i in range(r):
            if self.leaf_prediction == TARGET_MEAN:
                votes = self.get_votes_for_instance(X[i]).copy()
                # Tree is not empty, otherwise, all target_values are set
                # equally, default to zero
                if votes != {}:
                    number_of_examples_seen = votes[0]
                    sum_of_values = votes[1]
                    predictions[i] = sum_of_values / number_of_examples_seen
            elif self.leaf_prediction == PERCEPTRON:
                if self.examples_seen > 1:
                    normalized_sample = self.normalize_sample(X[i])
                    perceptron_weights = self.get_weights_for_instance(X[i])

                    normalized_base_prediction = perceptron_weights[0] @ \
                        normalized_sample
                    normalized_meta_prediction = perceptron_weights[1] @ \
                        np.append(normalized_base_prediction, 1.0)
                    mean = self.sum_of_values / self.examples_seen
                    sd = np.sqrt((self.sum_of_squares -
                                 (self.sum_of_values ** 2) /
                                 self.examples_seen) / self.examples_seen)
                    # Samples are normalized using just one sd, as proposed in
                    # the iSoup-Tree method
                    predictions[i] = normalized_meta_prediction * sd + mean
            elif self.leaf_prediction == ADAPTIVE:
                if self.examples_seen > 1:
                    # Mean predictor
                    votes = self.get_votes_for_instance(X[i]).copy()
                    number_of_examples_seen = votes[0]
                    sum_of_values = votes[1]
                    pred_M = sum_of_values / number_of_examples_seen

                    # Standard perceptron
                    normalized_sample = self.normalize_sample(X[i])
                    perceptron_weights = self.get_weights_for_instance(X[i])

                    normalized_base_prediction = perceptron_weights[0] @ \
                        normalized_sample
                    normalized_meta_prediction = perceptron_weights[1] @ \
                        np.append(normalized_base_prediction, 1.0)

                    mean = self.sum_of_values / self.examples_seen
                    sd = np.sqrt((self.sum_of_squares -
                                 (self.sum_of_values ** 2) /
                                 self.examples_seen) / self.examples_seen)

                    pred_P = normalized_base_prediction * sd + mean
                    pred_SP = normalized_meta_prediction * sd + mean

                    # Gets faded errors for the related leaf predictors
                    fmae = self._get_predictors_faded_error(X[i])

                    # Selects, for each target, the best current performer
                    for j in range(self._n_targets):
                        b_pred = np.argmin([fmae['mean'][j],
                                            fmae['perceptron'][j],
                                            fmae['stacked_perceptron'][j]]
                                           )

                        if b_pred == 0:
                            # If all the expected errors are the same,
                            # use the standard perceptron
                            if fmae['mean'][j] == fmae['perceptron'][j] \
                                    == fmae['stacked_perceptron'][j]:
                                predictions[i, j] = pred_P[j]
                            # Otherwise, use the simplest approach
                            else:
                                predictions[i, j] = pred_M[j]
                        else:
                            if b_pred == 1:
                                # Use the stacked perceptron if its expected
                                # error is the same than the error for the
                                # standard perceptron
                                if fmae['perceptron'][j] == \
                                        fmae['stacked_perceptron'][j]:
                                    predictions[i, j] = pred_SP[j]
                                else:
                                    predictions[i, j] = pred_P[j]
                            else:
                                predictions[i, j] = pred_SP[j]

        return predictions

    def _attempt_to_split(self, node, parent, parent_idx: int):
        """Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than
        the Hoeffding bound:
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
        split_criterion = IntraClusterVarianceReductionSplitCriterion()
        best_split_suggestions = node.\
            get_best_split_suggestions(split_criterion, self)

        best_split_suggestions.sort(key=attrgetter('merit'))
        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            hoeffding_bound = self.compute_hoeffding_bound(
                split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()
                ), self.split_confidence, node.get_weight_seen())
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]

            if (second_best_suggestion.merit / best_suggestion.merit <
                    1 - hoeffding_bound or hoeffding_bound <
                    self.tie_threshold):
                should_split = True
            if self.remove_poor_atts is not None and self.remove_poor_atts \
                    and not should_split:
                poor_atts = set()
                best_ratio = second_best_suggestion.merit \
                    / best_suggestion.merit

                # Add any poor attribute to set
                for i in range(len(best_split_suggestions)):
                    if best_split_suggestions[i].split_test is not None:
                        split_atts = best_split_suggestions[i].\
                            split_test.get_atts_test_depends_on()
                        if len(split_atts) == 1:
                            if best_split_suggestions[i].merit / \
                                    best_suggestion.merit < \
                                    best_ratio - 2 * hoeffding_bound:
                                poor_atts.add(int(split_atts[0]))

                for poor_att in poor_atts:
                    node.disable_attribute(poor_att)
        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                # Preprune - null wins
                self._deactivate_learning_node(node, parent, parent_idx)
            else:
                new_split = self.new_split_node(
                    split_decision.split_test,
                    node.get_observed_class_distribution()
                )
                for i in range(split_decision.num_splits()):
                    if self.leaf_prediction == PERCEPTRON:
                        new_child = self._new_learning_node(
                            split_decision.
                            resulting_class_distribution_from_split(i),
                            node.perceptron_weight
                        )
                    elif self.leaf_prediction == TARGET_MEAN:
                        new_child = self._new_learning_node(
                            split_decision.
                            resulting_class_distribution_from_split(i),
                            None)
                    elif self.leaf_prediction == ADAPTIVE:
                        new_child = self._new_learning_node(
                            split_decision.
                            resulting_class_distribution_from_split(i),
                            node.perceptron_weight
                        )
                        # Resets faded errors
                        new_child.fMAE_M = np.zeros(self._n_targets,
                                                    dtype=np.float64)
                        new_child.fMAE_P = np.zeros(self._n_targets,
                                                    dtype=np.float64)
                        new_child.fMAE_SP = np.zeros(self._n_targets,
                                                     dtype=np.float64)

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

    def _deactivate_learning_node(self, to_deactivate:
                                  RegressionHoeffdingTree.ActiveLearningNode,
                                  parent: RegressionHoeffdingTree.SplitNode,
                                  parent_branch: int):
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

        if self.leaf_prediction == TARGET_MEAN:
            new_leaf = self.InactiveLearningNodeForRegression(
                to_deactivate.get_observed_class_distribution()
            )
        elif self.leaf_prediction == PERCEPTRON:
            new_leaf = self.InactiveLearningNodePerceptron(
                to_deactivate.get_observed_class_distribution(),
                to_deactivate.perceptron_weight
            )
        elif self.leaf_prediction == ADAPTIVE:
            new_leaf = self.InactiveLearningNodeAdaptive(
                to_deactivate.get_observed_class_distribution(),
                to_deactivate.perceptron_weight
            )
            new_leaf.fMAE_M = to_deactivate.fMAE_M
            new_leaf.fMAE_P = to_deactivate.fMAE_P
            new_leaf.fMAE_SP = to_deactivate.fMAE_SP
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1
