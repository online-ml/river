import numpy as np

from skmultiflow.trees import RegressionHoeffdingTree

from skmultiflow.trees.nodes import AdaSplitNodeForRegression
from skmultiflow.trees.nodes import AdaLearningNodeForRegression


_TARGET_MEAN = 'mean'
_PERCEPTRON = 'perceptron'
ERROR_WIDTH_THRESHOLD = 300


class RegressionHAT(RegressionHoeffdingTree):
    """ An adaptation of the Hoeffding Adaptive Tree for regression.

    The tree uses ADWIN to detect drift and PERCEPTRON to make predictions.

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

    # ========================================================
    # == Regression Hoeffding Adaptive Tree implementation ===
    # ========================================================

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
            print("Invalid leaf_prediction option {}', will use default '{}'".format(leaf_prediction, _PERCEPTRON))
            self._leaf_prediction = _PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    def _new_learning_node(self, initial_class_observations=None, perceptron_weight=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}

        return AdaLearningNodeForRegression(initial_class_observations, perceptron_weight,
                                            random_state=self.random_state)

    def _partial_fit(self, X, y, weight):
        """Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: array_like
            Target value for sample X.
        weight: float
            Sample weight.

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
        return AdaSplitNodeForRegression(split_test, class_observations)
