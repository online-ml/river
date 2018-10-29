from skmultiflow.trees.regression_hoeffding_tree import RegressionHoeffdingTree, HoeffdingTree
from skmultiflow.utils.utils import *
import logging
from skmultiflow.drift_detection.adwin import ADWIN
from abc import ABCMeta, abstractmethod
from skmultiflow.utils import check_random_state


_TARGET_MEAN = 'mean'
_PERCEPTRON = 'perceptron'
error_width_threshold = 300
SplitNode = RegressionHoeffdingTree.SplitNode
LearningNodePerceptron = RegressionHoeffdingTree.LearningNodePerceptron

# logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionHAT(RegressionHoeffdingTree):
    """
    Regression Hoeffding trees known as Fast incremental model tree with drift detection (FIMT-DD).

    The tree uses ADWIN to detect drift and PERCEPTRON to make predictions

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
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'mean' - Target mean
        | 'perceptron' - Perceptron
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    learning_ratio_perceptron: flaot
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

    """

    class NewNode(metaclass=ABCMeta):
        """
            Abstract Class to create a New Node for RegressionHoeffdingAdaptiveTree (RegressionHAT)
        """

        @abstractmethod
        def number_leaves(self):
            pass

        @abstractmethod
        def get_error_estimation(self):
            pass

        @abstractmethod
        def get_error_width(self):
            pass

        @abstractmethod
        def is_null_error(self):
            pass

        @abstractmethod
        def kill_tree_children(self, hat):
            pass

        @abstractmethod
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
            pass

        @abstractmethod
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counts, found_nodes=None):
            pass

    class AdaSplitNodeForRegression(SplitNode, NewNode):
        def __init__(self, split_test, class_observations):
            super().__init__(split_test, class_observations)
            self._estimation_error_weight = ADWIN()
            self._alternate_tree = None
            self.error_change = False
            self._random_seed = 1
            self._classifier_random = check_random_state(self._random_seed)

        # Override NewNode
        def number_leaves(self):
            num_of_leaves = 0
            for child in self._children:
                if child is not None:
                    num_of_leaves += child.number_leaves()

            return num_of_leaves

        # Override NewNode
        def get_error_estimation(self):
            return self._estimation_error_weight.estimation

        # Override NewNode
        def get_error_width(self):
            w = 0.0
            if self.is_null_error() is False:
                w = self._estimation_error_weight.width

            return w

        # Override NewNode
        def is_null_error(self):
            return self._estimation_error_weight is None

        # Override NewNode
        def learn_from_instance(self, X, y, weight, rhat, parent, parent_branch):

            true_target = y

            normalized_error = 0.0

            if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
                target_prediction = rhat.predict([X])[0]
                normalized_error = rhat.get_normalized_error(target_prediction, true_target)
            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()

            # Add element to Change detector
            self._estimation_error_weight.add_element(normalized_error)

            # Detect change
            self.error_change = self._estimation_error_weight.detected_change()

            if self.error_change is True and old_error > self.get_error_estimation():

                self.error_change = False

            # Check condition to build a new alternate tree
            if self.error_change is True:
                self._alternate_tree = rhat._new_learning_node()
                rhat.alternate_trees_cnt += 1

            # Condition to replace alternate tree
            elif self._alternate_tree is not None and self._alternate_tree.is_null_error() is False:
                print("we'll be replacing the actual tree")
                if self.get_error_width() > error_width_threshold \
                        and self._alternate_tree.get_error_width() > error_width_threshold:
                    old_error_rate = self.get_error_estimation()
                    alt_error_rate = self._alternate_tree.get_error_estimation()
                    fDelta = .05
                    fN = 1.0 / self._alternate_tree.get_error_width() + 1.0 / (self.get_error_width())

                    bound = math.sqrt(2.0 * old_error_rate * (1.0 - old_error_rate) * math.log(2.0 / fDelta) * fN)
                    # To check, bound never less than (old_error_rate - alt_error_rate)
                    if bound < (old_error_rate - alt_error_rate):
                        rhat._active_leaf_node_cnt -= self.number_leaves()
                        rhat._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                        self.kill_tree_children(rhat)

                        if parent is not None:
                            parent.set_child(parent_branch, self._alternate_tree)
                        else:
                            rhat._tree_root = rhat._tree_root._alternate_tree
                        rhat.switch_alternate_trees_cnt += 1
                    elif bound < alt_error_rate - old_error_rate:
                        if isinstance(self._alternate_tree, HoeffdingTree.ActiveLearningNode):
                            self._alternate_tree = None
                        elif isinstance(self._alternate_tree, HoeffdingTree.ActiveLearningNode):
                            self._alternate_tree = None
                        else:
                            self._alternate_tree.kill_tree_children(rhat)
                        rhat.pruned_alternate_trees_cnt += 1  # hat.pruned_alternate_trees_cnt to check

            # Learn_From_Instance alternate Tree and Child nodes
            if self._alternate_tree is not None:
                self._alternate_tree.learn_from_instance(X, y, weight, rhat, parent, parent_branch)
            child_branch = self.instance_child_index(X)
            child = self.get_child(child_branch)
            if child is not None:
                child.learn_from_instance(X, y, weight, rhat, parent, parent_branch)

        # Override NewNode
        def kill_tree_children(self, rhat):
            for child in self._children:
                if child is not None:
                    # Delete alternate tree if it exists
                    if isinstance(child, rhat.AdaSplitNodeForRegression) and child._alternate_tree is not None:
                        self._pruned_alternate_trees += 1
                    # Recursive delete of SplitNodes
                    if isinstance(child, rhat.AdaSplitNodeForRegression):
                        child.kill_tree_children(rhat)

                    if isinstance(child, HoeffdingTree.ActiveLearningNode):
                        child = None
                        rhat._active_leaf_node_cnt -= 1
                    elif isinstance(child, HoeffdingTree.InactiveLearningNode):
                        child = None
                        rhat._inactive_leaf_node_cnt -= 1

        # override NewNode
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counts=False, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
            if update_splitter_counts:

                try:

                    self._observed_class_distribution[0] += weight
                    self._observed_class_distribution[1] += y * weight
                    self._observed_class_distribution[2] += y * y * weight

                except KeyError:

                    self._observed_class_distribution[0] = weight
                    self._observed_class_distribution[1] = y * weight
                    self._observed_class_distribution[2] = y * y * weight

            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    child.filter_instance_to_leaves(X, y, weight, parent, parent_branch,
                                                    update_splitter_counts, found_nodes)
                else:
                    found_nodes.append(HoeffdingTree.FoundNode(None, self, child_index))
            if self._alternate_tree is not None:
                self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999,
                                                               update_splitter_counts, found_nodes)

    class AdaLearningNodeForRegression(LearningNodePerceptron, NewNode):

        def __init__(self, initial_class_observations, perceptron_weight, random_state=None):
            super().__init__(initial_class_observations, perceptron_weight, random_state)
            self._estimation_error_weight = ADWIN()
            self._error_change = False
            self._randomSeed = 1
            self._classifier_random = check_random_state(self._randomSeed)

        # Override NewNode
        def number_leaves(self):
            return 1

        # Override NewNode
        def get_error_estimation(self):
            return self._estimation_error_weight.estimation

        # Override NewNode
        def get_error_width(self):
            return self._estimation_error_weight.width

        # Override NewNode
        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_children(self, hat):
            pass

        # Override NewNode
        def learn_from_instance(self, X, y, weight, rhat, parent, parent_branch):

            super().learn_from_instance(X, y, weight, rhat)

            true_target = y
            target_prediction = rhat.predict([X])[0]

            normalized_error = rhat.get_normalized_error(target_prediction, true_target)

            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()

            # Add element to Adwin

            self._estimation_error_weight.add_element(normalized_error)
            # Detect change with Adwin
            self._error_change = self._estimation_error_weight.detected_change()

            if self._error_change is True and old_error > self.get_error_estimation():
                self._error_change = False

            # call ActiveLearningNode
            weight_seen = self.get_weight_seen()

            if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= rhat.grace_period:
                rhat._attempt_to_split(self, parent, parent_branch)
                self.set_weight_seen_at_last_split_evaluation(weight_seen)

        # Override NewNode, New for option votes
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counts, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
            found_nodes.append(HoeffdingTree.FoundNode(self, parent, parent_branch))

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

        super(RegressionHAT, self).__init__(max_byte_size=max_byte_size,
                                            memory_estimate_period=memory_estimate_period,
                                            grace_period=grace_period,
                                            split_confidence=split_confidence,
                                            tie_threshold=tie_threshold,
                                            binary_split=binary_split,
                                            stop_mem_management=stop_mem_management,
                                            remove_poor_atts=remove_poor_atts,
                                            no_preprune=no_preprune,
                                            nb_threshold=nb_threshold,
                                            nominal_attributes=nominal_attributes,
                                            learning_ratio_perceptron=learning_ratio_perceptron,
                                            learning_ratio_decay=learning_ratio_decay,
                                            learning_ratio_const=learning_ratio_const,
                                            leaf_prediction=leaf_prediction,
                                            random_state=random_state)
        self.alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {_TARGET_MEAN, _PERCEPTRON}:
            logger.info("Invalid option {}', will use default '{}'".format(leaf_prediction, _PERCEPTRON))
            self._leaf_prediction = _PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    def _new_learning_node(self, initial_class_observations=None, perceptron_weight=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}

        return self.AdaLearningNodeForRegression(initial_class_observations, perceptron_weight,
                                                 random_state=self._init_random_state)

    def _partial_fit(self, X, y, weight):
        """Trains the model on samples X and corresponding targets y.

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

        self.samples_seen += weight
        self.sum_of_values += weight * y
        self.sum_of_squares += weight * y * y

        try:
            self.sum_of_attribute_values = np.add(self.sum_of_attribute_values, np.multiply(weight, X))
            self.sum_of_attribute_squares = np.add(self.sum_of_attribute_squares, np.multiply(weight, np.power(X, 2)))
        except ValueError:
            self.sum_of_attribute_values = np.multiply(weight, X)
            self.sum_of_attribute_squares = np.multiply(weight, np.power(X, 2))

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        self._tree_root.learn_from_instance(X, y, weight, self, None, -1)

    def get_normalized_error(self, prediction, y):
        normal_prediction = self.normalized_target_value(prediction)
        normal_value = self.normalized_target_value(y)
        return np.abs(normal_value-normal_prediction)

    def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, split_parent, parent_branch,
                                                  update_splitter_counts, nodes)
        return nodes

    def new_split_node(self, split_test, class_observations):
        return self.AdaSplitNodeForRegression(split_test, class_observations)
