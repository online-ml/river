import numpy as np
from operator import attrgetter

from skmultiflow.core import RegressorMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions

from ._split_criterion import VarianceReductionSplitCriterion
from ._attribute_test import NominalAttributeMultiwayTest
from ._nodes import SplitNode
from ._nodes import LearningNode
from ._nodes import ActiveLeaf, InactiveLeaf
from ._nodes import ActiveLearningNodeMean
from ._nodes import InactiveLearningNodeMean
from ._nodes import ActiveLearningNodePerceptron
from ._nodes import InactiveLearningNodePerceptron
from ._nodes.htr_nodes import compute_sd

import warnings


def RegressionHoeffdingTree(max_byte_size=33554432, memory_estimate_period=1000000,
                            grace_period=200, split_confidence=0.0000001, tie_threshold=0.05,
                            binary_split=False, stop_mem_management=False, remove_poor_atts=False,
                            leaf_prediction="perceptron", no_preprune=False, nb_threshold=0,
                            nominal_attributes=None, learning_ratio_perceptron=0.02,
                            learning_ratio_decay=0.001, learning_ratio_const=True,
                            random_state=None):     # pragma: no cover
    warnings.warn("'.RegressionHoeffdingTree' has been renamed to 'HoeffdingTreeRegressor' in "
                  "v0.5.0.\nThe old name will be removed in v0.7.0", category=FutureWarning)
    return HoeffdingTreeRegressor(max_byte_size=max_byte_size,
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


class HoeffdingTreeRegressor(RegressorMixin, HoeffdingTreeClassifier):
    """ Hoeffding Tree regressor.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
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
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
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

    Notes
    -----
    The Hoeffding Tree Regressor (HTR) is an adaptation of the incremental tree algorithm of the
    same name for classification. Similarly to its classification counterpart, HTR uses the
    Hoeffding bound to control its split decisions. Differently from the classification algorithm,
    HTR relies on calculating the reduction of variance in the target space to decide among the
    split candidates. The smallest the variance at its leaf nodes, the more homogeneous the
    partitions are. At its leaf nodes, HTR fits either linear perceptron models or uses the sample
    average as the predictor.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import RegressionGenerator
    >>> from skmultiflow.trees import HoeffdingTreeRegressor
    >>> import numpy as np
    >>>
    >>> # Setup a data stream
    >>> stream = RegressionGenerator(random_state=1, n_samples=200)
    >>>
    >>> # Setup the Hoeffding Tree Regressor
    >>> ht_reg = HoeffdingTreeRegressor()
    >>>
    >>> # Auxiliary variables to control loop and track performance
    >>> n_samples = 0
    >>> max_samples = 200
    >>> y_pred = np.zeros(max_samples)
    >>> y_true = np.zeros(max_samples)
    >>>
    >>> # Run test-then-train loop for max_samples and while there is data
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_true[n_samples] = y[0]
    >>>     y_pred[n_samples] = ht_reg.predict(X)[0]
    >>>     ht_reg.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Hoeffding Tree regressor mean absolute error: {}'.
    >>>       format(np.mean(np.abs(y_true - y_pred))))
    """

    _TARGET_MEAN = 'mean'
    _PERCEPTRON = 'perceptron'

    # =============================================
    # == Hoeffding Tree Regressor implementation ==
    # =============================================
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
                         nominal_attributes=nominal_attributes,
                         split_criterion='vr')
        self.split_criterion = 'vr'   # variance reduction

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
        self.samples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0
        self.sum_of_attribute_values = []
        self.sum_of_attribute_squares = []
        self.random_state = random_state

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {self._TARGET_MEAN, self._PERCEPTRON}:
            print("Invalid leaf_prediction option {}', will use default '{}'".
                  format(leaf_prediction, self._PERCEPTRON))
            self._leaf_prediction = self._PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != 'vr':   # variance reduction
            print("Invalid split_criterion option {}', will use default '{}'".
                  format(split_criterion, 'vr'))
            self._split_criterion = 'vr'
        else:
            self._split_criterion = split_criterion

    def normalize_sample(self, X):
        """
        Normalize the features in order to have the same influence during training.

        Parameters
        ----------
        X: list or array or numpy.ndarray
            features.
        Returns
        -------
        array:
            normalized samples
        """
        normalized_sample = []
        for i in range(len(X)):
            if (self.nominal_attributes is None or (self.nominal_attributes is not None and
                                                    i not in self.nominal_attributes)) and \
                    self.samples_seen > 1:
                mean = self.sum_of_attribute_values[i] / self.samples_seen
                sd = compute_sd(self.sum_of_attribute_squares[i], self.sum_of_attribute_values[i],
                                self.samples_seen)
                if sd > 0:
                    normalized_sample.append(float(X[i] - mean) / (3 * sd))
                else:
                    normalized_sample.append(0.0)
            elif self.nominal_attributes is not None and i in self.nominal_attributes:
                normalized_sample.append(X[i])  # keep nominal inputs unaltered
            else:
                normalized_sample.append(0.0)
        if self.samples_seen > 1:
            normalized_sample.append(1.0)  # Value to be multiplied with the constant factor
        else:
            normalized_sample.append(0.0)
        return np.asarray(normalized_sample)

    def normalize_target_value(self, y):
        """
        Normalize the target in order to have the same influence during training.

        Parameters
        ----------
        y: float
            target value

        Returns
        -------
        float
            normalized target value
        """
        if self.samples_seen > 1:
            mean = self.sum_of_values / self.samples_seen
            sd = compute_sd(self.sum_of_squares, self.sum_of_values, self.samples_seen)
            if sd > 0:
                return float(y - mean) / (3 * sd)
        return 0.0

    def _new_learning_node(self, initial_stats=None, parent_node=None, is_active=True):
        """Create a new learning node. The type of learning node depends on the tree
        configuration."""
        if initial_stats is None:
            initial_stats = {}

        if is_active:
            if self.leaf_prediction == self._TARGET_MEAN:
                return ActiveLearningNodeMean(initial_stats)
            elif self.leaf_prediction == self._PERCEPTRON:
                return ActiveLearningNodePerceptron(initial_stats, parent_node,
                                                    random_state=self.random_state)
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats)
            elif self.leaf_prediction == self._PERCEPTRON:
                return InactiveLearningNodePerceptron(initial_stats, parent_node,
                                                      random_state=self.random_state)

    def partial_fit(self, X, y, sample_weight=None):
        """ Incrementally trains the model.

        Train samples (instances) are composed of X attributes and their corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for the instance and
          update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has observed between split
          attempts exceed the grace period then attempt to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Target values for all samples in X.
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        """
        if y is not None:
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
        y: float
            Target value for sample X.
        sample_weight: float
            Samples weight.

        """

        self.samples_seen += sample_weight
        self.sum_of_values += sample_weight * y
        self.sum_of_squares += sample_weight * y * y

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
        # (in a multi-way test)
        elif isinstance(leaf_node, SplitNode) and \
                isinstance(leaf_node.split_test, NominalAttributeMultiwayTest):
            current = found_node.node
            leaf_node = self._new_learning_node()
            branch_id = current.split_test.add_new_branch(
                X[current.split_test.get_atts_test_depends_on()[0]]
            )
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
        numpy.ndarray
            Predicted target values.

        """
        predictions = []
        if self.samples_seen > 0 and self._tree_root is not None:
            r, _ = get_dimensions(X)
            for i in range(r):
                node = self._tree_root.filter_instance_to_leaf(X[i], None, -1).node
                if node.is_leaf():
                    predictions.append(node.predict_one(X[i], tree=self))
                else:
                    # The instance sorting ended up in a Split Node, since no branch was found
                    # for some of the instance's features. Use the mean prediction in this case
                    predictions.append(node.stats[1] / node.stats[0])
        else:
            # Model is empty
            predictions.append(0.0)
        return np.asarray(predictions)

    def predict_proba(self, X):
        """Not implemented for this method
        """
        raise NotImplementedError

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
        node:
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
            hoeffding_bound = self._hoeffding_bound(
                split_criterion.get_range_of_merit(node.stats), self.split_confidence,
                node.total_weight)
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]
            if best_suggestion.merit > 0.0 and \
                    (second_best_suggestion.merit / best_suggestion.merit < 1 - hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):
                should_split = True
            if self.remove_poor_atts:
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

    def _sort_learning_nodes(self, learning_nodes):
        """ Define strategy to sort learning nodes according to their likeliness of being split."""
        learning_nodes.sort(key=lambda n: n.depth, reverse=True)
        return learning_nodes

    def _activate_learning_node(self, to_activate: InactiveLeaf, parent: SplitNode,
                                parent_branch: int):
        """ Activate a learning node.

        Parameters
        ----------
        to_activate: InactiveLeaf
            The node to activate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(
            to_activate.stats, to_activate
        )
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def _deactivate_learning_node(self, to_deactivate: ActiveLeaf,
                                  parent: SplitNode, parent_branch: int):
        """Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: ActiveLeaf
            The node to deactivate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(to_deactivate.stats,
                                           to_deactivate,
                                           is_active=False)

        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1
