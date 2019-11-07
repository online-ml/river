import numpy as np
from operator import attrgetter

from skmultiflow.core import RegressorMixin
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.utils import *
from skmultiflow.utils import check_random_state
from skmultiflow.trees.split_criterion import VarianceReductionSplitCriterion

from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees.nodes import LearningNode
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import ActiveLearningNodeForRegression
from skmultiflow.trees.nodes import InactiveLearningNodeForRegression
from skmultiflow.trees.nodes import ActiveLearningNodePerceptron
from skmultiflow.trees.nodes import InactiveLearningNodePerceptron

_TARGET_MEAN = 'mean'
_PERCEPTRON = 'perceptron'


class RegressionHoeffdingTree(RegressorMixin, HoeffdingTree):
    """ Regression Hoeffding Tree or Fast Incremental Model Tree with Drift Detection.

    This is an implementation of the Fast Incremental Model Tree with Drift Detection (FIMT-DD)
    introduced by E. Ikonomovska, J. Gama, and S. Džeroski [1]_.

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

    References
    ----------
    .. [1] Elena Ikonomovska, João Gama, and Sašo Džeroski. 2011. Learning model trees from
           evolving data streams. Data Min. Knowl. Discov. 23, 1 (July 2011), 128-168.
    """

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
        if leaf_prediction not in {_TARGET_MEAN, _PERCEPTRON}:
            print("Invalid leaf_prediction option {}', will use default '{}'".format(leaf_prediction, _PERCEPTRON))
            self._leaf_prediction = _PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != 'vr':   # variance reduction
            print("Invalid split_criterion option {}', will use default '{}'".format(split_criterion,
                                                                                     'vr'))
            self._split_criterion = 'vr'
        else:
            self._split_criterion = split_criterion

    def normalize_sample(self, X):
        """
        Normalize the features in order to have the same influence during
        training.

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
            mean = self.sum_of_attribute_values[i] / self.samples_seen
            sd = np.sqrt((self.sum_of_attribute_squares[i] - self.sum_of_attribute_values[i] ** 2
                          / self.samples_seen) / self.samples_seen)
            if self.samples_seen > 1 and sd > 0:
                normalized_sample.append((X[i] - mean) / (3 * sd))
            else:
                normalized_sample.append(0.0)
        if self.samples_seen > 1:
            normalized_sample.append(1.0)  # Value to be multiplied with the constant factor
        else:
            normalized_sample.append(0.0)
        return normalized_sample

    def normalized_target_value(self, y):
        """
        Normalize the target in order to have the same influence during the process of
        training.
        Parameters
        ----------
        y: float
            target value

        Returns
        -------
        float:
            normalized target value
        """
        if self.samples_seen > 1:
            mean = self.sum_of_values / self.samples_seen
            sd = np.sqrt((self.sum_of_squares - self.sum_of_values ** 2
                          / self.samples_seen) / self.samples_seen)
            if sd > 0:
                return (y - mean) / (3 * sd)
            else:
                return 0.0
        return 0.0

    def _new_learning_node(self, initial_class_observations=None, perceptron_weight=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self.leaf_prediction == _TARGET_MEAN:
            return ActiveLearningNodeForRegression(initial_class_observations)
        elif self.leaf_prediction == _PERCEPTRON:
            return ActiveLearningNodePerceptron(initial_class_observations, perceptron_weight,
                                                random_state=self.random_state)

    def get_weights_for_instance(self, X):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.perceptron_weight
        else:
            return []

    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally trains the model. Train samples (instances) are composed of X attributes and their
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
            Target values for all samples in X.
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        """
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt,
                                                                                                  len(sample_weight)))
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
            self.sum_of_attribute_values = np.add(self.sum_of_attribute_values, np.multiply(sample_weight, X))
            self.sum_of_attribute_squares = np.add(
                self.sum_of_attribute_squares, np.multiply(sample_weight, np.power(X, 2))
            )
        except ValueError:
            self.sum_of_attribute_values = np.multiply(sample_weight, X)
            self.sum_of_attribute_squares = np.multiply(sample_weight, np.power(X, 2))

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
            learning_node.learn_from_instance(X, y, sample_weight, self)
            if self._growth_allowed and isinstance(learning_node, ActiveLearningNode):
                active_learning_node = learning_node
                weight_seen = active_learning_node.get_weight_seen()
                weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                if weight_diff >= self.grace_period:
                    self._attempt_to_split(active_learning_node, found_node.parent, found_node.parent_branch)
                    active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()

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
        if self.samples_seen > 0:
            r, _ = get_dimensions(X)
            for i in range(r):
                if self.leaf_prediction == _TARGET_MEAN:
                    votes = self.get_votes_for_instance(X[i]).copy()
                    if votes == {}:
                        # Tree is empty, all target_values equal, default to zero
                        predictions.append(0)
                    else:
                        number_of_samples_seen = votes[0]
                        sum_of_values = votes[1]
                        predictions.append(sum_of_values / number_of_samples_seen)
                elif self.leaf_prediction == _PERCEPTRON:
                    normalized_sample = self.normalize_sample(X[i])
                    normalized_prediction = np.dot(self.get_weights_for_instance(X[i]), normalized_sample)
                    mean = self.sum_of_values / self.samples_seen
                    sd = np.sqrt((self.sum_of_squares - self.sum_of_values ** 2 / self.samples_seen)
                                 / self.samples_seen)
                    if self.samples_seen > 1:
                        predictions.append(normalized_prediction * sd * 3 + mean)
                    else:
                        predictions.append(0.0)
        else:
            # Model is empty
            predictions.append(0.0)
        return np.asarray(predictions)

    def predict_proba(self, X):
        """Not implemented for this method
        """
        raise NotImplementedError

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
            if best_suggestion.merit > 0.0 and \
                    (second_best_suggestion.merit / best_suggestion.merit < 1 - hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):
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
                    if self.leaf_prediction == _PERCEPTRON:
                        new_child = self._new_learning_node(
                            split_decision.resulting_class_distribution_from_split(i),
                            node.perceptron_weight
                        )
                    else:
                        new_child = self._new_learning_node(
                            split_decision.resulting_class_distribution_from_split(i),
                            None
                        )
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

    def _deactivate_learning_node(self, to_deactivate: ActiveLearningNode,
                                  parent: SplitNode, parent_branch: int):
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
        if self.leaf_prediction == _TARGET_MEAN:
            new_leaf = InactiveLearningNodeForRegression(
                to_deactivate.get_observed_class_distribution()
            )
        else:
            new_leaf = InactiveLearningNodePerceptron(
                to_deactivate.get_observed_class_distribution(),
                to_deactivate.perceptron_weight
            )
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1
