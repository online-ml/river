from operator import attrgetter

import numpy as np

from skmultiflow.core import MultiOutputMixin
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.utils import get_dimensions

from ._split_criterion import IntraClusterVarianceReductionSplitCriterion
from ._attribute_test import NominalAttributeMultiwayTest
from ._nodes import SplitNode
from ._nodes import LearningNode
from ._nodes import ActiveLeaf
from ._nodes import ActiveLearningNodeMean
from ._nodes import ActiveLearningNodePerceptronMultiTarget
from ._nodes import ActiveLearningNodeAdaptiveMultiTarget
from ._nodes import InactiveLearningNodeMean
from ._nodes import InactiveLearningNodePerceptronMultiTarget
from ._nodes import InactiveLearningNodeAdaptiveMultiTarget

import warnings


def MultiTargetRegressionHoeffdingTree(max_byte_size=33554432, memory_estimate_period=1000000,
                                       grace_period=200, split_confidence=0.0000001,
                                       tie_threshold=0.05, binary_split=False,
                                       stop_mem_management=False, remove_poor_atts=False,
                                       leaf_prediction='perceptron', no_preprune=False,
                                       nb_threshold=0, nominal_attributes=None,
                                       learning_ratio_perceptron=0.02, learning_ratio_decay=0.001,
                                       learning_ratio_const=True,
                                       random_state=None):     # pragma: no cover
    warnings.warn("'MultiTargetRegressionHoeffdingTree' has been renamed to 'iSOUPTreeRegressor'"
                  "in v0.5.0.\nThe old name will be removed in v0.7.0", category=FutureWarning)
    return iSOUPTreeRegressor(max_byte_size=max_byte_size,
                              memory_estimate_period=memory_estimate_period,
                              grace_period=grace_period,
                              split_confidence=split_confidence,
                              tie_threshold=tie_threshold,
                              binary_split=binary_split,
                              stop_mem_management=stop_mem_management,
                              remove_poor_atts=remove_poor_atts,
                              leaf_prediction=leaf_prediction,
                              no_preprune=no_preprune,
                              nb_threshold=nb_threshold,
                              nominal_attributes=nominal_attributes,
                              learning_ratio_perceptron=learning_ratio_perceptron,
                              learning_ratio_decay=learning_ratio_decay,
                              learning_ratio_const=learning_ratio_const,
                              random_state=random_state)


class iSOUPTreeRegressor(HoeffdingTreeRegressor, MultiOutputMixin):
    """ Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target regression.

    This is an implementation of the iSOUP-Tree proposed by A. Osojnik, P. Panov, and
    S. Džeroski [1]_.

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
    leaf_prediction: string (default='perceptron')
        | Prediction mechanism used at leafs.
        | 'mean' - Target mean
        | 'perceptron' - Perceptron
        | 'adaptive' - Adaptively chooses between the best predictor
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
    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Used when leaf_prediction is 'perceptron'.

    References
    ----------
    .. [1] Aljaž Osojnik, Panče Panov, and Sašo Džeroski. "Tree-based methods for online
        multi-target regression." Journal of Intelligent Information Systems 50.2 (2018): 315-339.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import RegressionGenerator
    >>> from skmultiflow.trees import iSOUPTreeRegressor
    >>> import numpy as np
    >>>
    >>> # Setup a data stream
    >>> n_targets = 3
    >>> stream = RegressionGenerator(n_targets=n_targets, random_state=1, n_samples=200)
    >>>
    >>> # Setup iSOUP Tree Regressor
    >>> isoup_tree = iSOUPTreeRegressor()
    >>>
    >>> # Auxiliary variables to control loop and track performance
    >>> n_samples = 0
    >>> max_samples = 200
    >>> y_pred = np.zeros((max_samples, n_targets))
    >>> y_true = np.zeros((max_samples, n_targets))
    >>>
    >>> # Run test-then-train loop for max_samples and while there is data
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_true[n_samples] = y[0]
    >>>     y_pred[n_samples] = isoup_tree.predict(X)[0]
    >>>     isoup_tree.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('iSOUP Tree regressor example')
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))
    """

    _ADAPTIVE = 'adaptive'
    # ============================================================
    # == Multi-target Regression Hoeffding Tree implementation ===
    # ============================================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 leaf_prediction='perceptron',
                 no_preprune=False,
                 nb_threshold=0,
                 nominal_attributes=None,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True,
                 random_state=None):
        super().__init__(max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes)
        self.split_criterion = 'icvr'   # intra cluster variance reduction
        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const
        self.random_state = random_state

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

        self.examples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0
        self.sum_of_attribute_values = 0.0
        self.sum_of_attribute_squares = 0.0

        # To add the n_targets property once
        self._n_targets_set = False

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {self._TARGET_MEAN, self._PERCEPTRON, self._ADAPTIVE}:
            print("Invalid leaf_prediction option {}', will use default '{}'".format(
                leaf_prediction, self._PERCEPTRON))
            self._leaf_prediction = self._PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion == 'vr':
            # Corner case due to parent class initialization
            split_criterion = 'icvr'
        if split_criterion != 'icvr':   # intra cluster variance reduction
            print("Invalid split_criterion option {}', will use default '{}'"
                  .format(split_criterion, 'icvr'))
            self._split_criterion = 'icvr'
        else:
            self._split_criterion = split_criterion

    def normalize_sample(self, X):
        """Normalize the features in order to have the same influence during the
        process of training.

        Parameters
        ----------
        X: np.array
            features.
        Returns
        -------
        np.array:
            normalized samples
        """
        if self.examples_seen <= 1:
            _, c = get_dimensions(X)
            return np.zeros((c + 1), dtype=np.float64)

        mean = self.sum_of_attribute_values / self.examples_seen
        variance = ((self.sum_of_attribute_squares - (self.sum_of_attribute_values *
                     self.sum_of_attribute_values) / self.examples_seen)
                    / (self.examples_seen - 1))

        sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)

        normalized_sample = np.zeros(X.shape[0] + 1, dtype=np.float64)
        np.divide(X - mean, sd, where=sd != 0, out=normalized_sample[:-1])
        # Augments sample with the bias input signal (or y intercept for
        # each target)
        normalized_sample[-1] = 1.0

        return normalized_sample

    def normalize_target_value(self, y):
        """Normalize the targets in order to have the same influence during the
        process of training.

        Parameters
        ----------
        y: np.array
            targets.

        Returns
        -------
        np.array:
            normalized targets values
        """
        if self.examples_seen <= 1:
            return np.zeros_like(y, dtype=np.float64)

        mean = self.sum_of_values / self.examples_seen
        variance = ((self.sum_of_squares - (self.sum_of_values * self.sum_of_values)
                    / self.examples_seen) / (self.examples_seen - 1))

        sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)

        normalized_targets = np.divide(y - mean, sd, where=sd != 0,
                                       out=np.zeros_like(y, dtype=np.float64))

        return normalized_targets

    def _new_learning_node(self, initial_stats=None, parent_node=None,
                           is_active=True):
        """Create a new learning node. The type of learning node depends on
        the tree configuration.
        """
        if initial_stats is None:
            initial_stats = {}

        if is_active:
            if self.leaf_prediction == self._TARGET_MEAN:
                return ActiveLearningNodeMean(initial_stats)
            elif self.leaf_prediction == self._PERCEPTRON:
                return ActiveLearningNodePerceptronMultiTarget(
                    initial_stats, parent_node, random_state=self.random_state)
            elif self.leaf_prediction == self._ADAPTIVE:
                new_node = ActiveLearningNodeAdaptiveMultiTarget(
                    initial_stats, parent_node, random_state=self.random_state)
                # Resets faded errors
                new_node.fMAE_M = np.zeros(self._n_targets, dtype=np.float64)
                new_node.fMAE_P = np.zeros(self._n_targets, dtype=np.float64)
                return new_node
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats)
            elif self.leaf_prediction == self._PERCEPTRON:
                return InactiveLearningNodePerceptronMultiTarget(
                    initial_stats, parent_node, random_state=parent_node.random_state)
            elif self.leaf_prediction == self._ADAPTIVE:
                new_node = InactiveLearningNodeAdaptiveMultiTarget(
                    initial_stats, parent_node, random_state=parent_node.random_state)
                new_node.fMAE_M = parent_node.fMAE_M
                new_node.fMAE_P = parent_node.fMAE_P
                return new_node

    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally trains the model. Train samples (instances) are
        composed of X attributes and their corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are
          assumed.
        * If more than one instance is passed, loop through X and pass
          instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: numpy.ndarray of shape (n_samples, n_targets)
            Target values.
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.
        """
        if y is not None:
            # Set the number of targets once
            if not self._n_targets_set:
                _, self._n_targets = get_dimensions(y)
                self._n_targets_set = True

            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.
                                 format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self._partial_fit(X[i], y[i], sample_weight[i])

    def _partial_fit(self, X, y, sample_weight):
        """Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: array_like
            numpy.ndarray of shape (1, n_targets)
                Target values for sample X.
        sample_weight: float
            Sample weight.
        """
        try:
            self.examples_seen += sample_weight
            self.sum_of_values += sample_weight * y
            self.sum_of_squares += sample_weight * y * y
        except ValueError:
            self.examples_seen = sample_weight
            self.sum_of_values = sample_weight * y
            self.sum_of_squares = sample_weight * y * y

        try:
            self.sum_of_attribute_values += sample_weight * X
            self.sum_of_attribute_squares += sample_weight * X * X
        except ValueError:
            self.sum_of_attribute_values = sample_weight * X
            self.sum_of_attribute_squares = sample_weight * X * X

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1

        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node
        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._active_leaf_node_cnt += 1

        if isinstance(leaf_node, LearningNode):
            learning_node = leaf_node
            learning_node.learn_one(X, y, weight=sample_weight, tree=self)

            if self._growth_allowed and isinstance(learning_node, ActiveLeaf):
                active_learning_node = learning_node
                weight_seen = active_learning_node.total_weight

                weight_diff = weight_seen - active_learning_node.last_split_attempt_at
                if weight_diff >= self.grace_period:
                    self._attempt_to_split(active_learning_node, found_node.parent,
                                           found_node.parent_branch)
                    active_learning_node.last_split_attempt_at = weight_seen
        # Split node encountered a previously unseen categorical value
        # (in a multiway test)
        elif isinstance(leaf_node, SplitNode) and \
                isinstance(leaf_node.split_test, NominalAttributeMultiwayTest):
            current = found_node.node
            leaf_node = self._new_learning_node()
            branch_id = current.split_test.add_new_branch(
                X[current.split_test.get_atts_test_depends_on()[0]])
            current.set_child(branch_id, leaf_node)
            self._active_leaf_node_cnt += 1
            leaf_node.learn_one(X, y, weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_byte_size()

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
        try:
            predictions = np.zeros((r, self._n_targets), dtype=np.float64)
        except AttributeError:
            warnings.warn("Calling predict without previously fitting the model at least once.\n"
                          "Predictions will default to a column array filled with zeros.")
            return np.zeros((r, 1))
        for i in range(r):
            node = self._tree_root.filter_instance_to_leaf(X[i], None, -1).node

            if isinstance(node, SplitNode):
                # If not leaf, use mean as response
                predictions[i, :] = node.stats[1] / node.stats[0] if len(node.stats) > 0 else 0.0
                continue
            predictions[i, :] = node.predict_one(X[i], tree=self)

        return predictions

    def predict_proba(self, X):
        """Not implemented for this method
        """
        raise NotImplementedError

    def _attempt_to_split(self, node, parent, parent_idx: int):
        """Attempt to split a node.

        If there exists significant variance among the target space of the
        seem examples:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the merit ratio of the top 2 split
        candidates is smaller than 1 minus the Hoeffding bound:
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
            hoeffding_bound = self._hoeffding_bound(
                split_criterion.get_range_of_merit(
                    node.stats
                ), self.split_confidence, node.total_weight)
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]

            if (best_suggestion.merit > 0 and (second_best_suggestion.merit / best_suggestion.merit
                                               < 1 - hoeffding_bound or hoeffding_bound
                                               < self.tie_threshold)):
                should_split = True
            if self.remove_poor_atts and not should_split:
                poor_atts = set()
                best_ratio = second_best_suggestion.merit / best_suggestion.merit

                # Add any poor attribute to set
                for i in range(len(best_split_suggestions)):
                    if best_split_suggestions[i].split_test is not None:
                        split_atts = best_split_suggestions[i].split_test.\
                            get_atts_test_depends_on()
                        if len(split_atts) == 1:
                            if (best_split_suggestions[i].merit / best_suggestion.merit
                                    < best_ratio - 2 * hoeffding_bound):
                                poor_atts.add(int(split_atts[0]))

                for poor_att in poor_atts:
                    node.disable_attribute(poor_att)

        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                # Preprune - null wins
                self._deactivate_learning_node(node, parent, parent_idx)
            else:
                new_split = self._new_split_node(split_decision.split_test, node.stats)
                for i in range(split_decision.num_splits()):
                    new_child = self._new_learning_node(
                        split_decision.resulting_stats_from_split(i), node)
                    new_split.set_child(i, new_child)

                self._active_leaf_node_cnt -= 1
                self._decision_node_cnt += 1
                self._active_leaf_node_cnt += split_decision.num_splits()
                if parent is None:
                    self._tree_root = new_split
                else:
                    parent.set_child(parent_idx, new_split)
            # Manage memory
            self._enforce_tracker_limit()
        elif len(best_split_suggestions) >= 2 and best_split_suggestions[-1].merit > 0 and \
                best_split_suggestions[-2].merit > 0:
            last_check_ratio = best_split_suggestions[-2].merit / best_split_suggestions[-1].merit
            last_check_sdr = best_split_suggestions[-1].merit

            node.manage_memory(split_criterion, last_check_ratio, last_check_sdr, hoeffding_bound)

    def _more_tags(self):
        return {'multioutput': True,
                'multioutput_only': True}
