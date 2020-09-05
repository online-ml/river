import numpy as np

from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.utils import add_dict_values

from ._nodes import InactiveLeaf
from ._nodes import AdaSplitNodeRegressor
from ._nodes import AdaActiveLearningNodeRegressor
from ._nodes import InactiveLearningNodeMean, InactiveLearningNodePerceptron

import warnings


def RegressionHAT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                  split_confidence=0.0000001, tie_threshold=0.05, binary_split=False,
                  stop_mem_management=False, remove_poor_atts=False, leaf_prediction="perceptron",
                  no_preprune=False, nominal_attributes=None, learning_ratio_perceptron=0.02,
                  learning_ratio_decay=0.001, learning_ratio_const=True, bootstrap_sampling=False,
                  random_state=None):     # pragma: no cover
    warnings.warn("'RegressionHAT' has been renamed to 'HoeffdingAdaptiveTreeRegressor' in"
                  "v0.5.0.\nThe old name will be removed in v0.7.0", category=FutureWarning)
    return HoeffdingAdaptiveTreeRegressor(max_byte_size=max_byte_size,
                                          memory_estimate_period=memory_estimate_period,
                                          grace_period=grace_period,
                                          split_confidence=split_confidence,
                                          tie_threshold=tie_threshold,
                                          binary_split=binary_split,
                                          stop_mem_management=stop_mem_management,
                                          remove_poor_atts=remove_poor_atts,
                                          leaf_prediction=leaf_prediction,
                                          no_preprune=no_preprune,
                                          nominal_attributes=nominal_attributes,
                                          learning_ratio_perceptron=learning_ratio_perceptron,
                                          learning_ratio_decay=learning_ratio_decay,
                                          learning_ratio_const=learning_ratio_const,
                                          bootstrap_sampling=bootstrap_sampling,
                                          random_state=random_state)


class HoeffdingAdaptiveTreeRegressor(HoeffdingTreeRegressor):
    """ Hoeffding Adaptive Tree regressor.

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
    leaf_prediction: string (default='perceptron')
        | Prediction mechanism used at leafs.
        | 'mean' - Target mean
        | 'perceptron' - Perceptron
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    learning_ratio_perceptron: float
        The learning rate of the perceptron.
    learning_ratio_decay: float
        Decay multiplier for the learning rate of the perceptron
    learning_ratio_const: Bool
        If False the learning ratio will decay with the number of examples seen
    bootstrap_sampling: bool, optional (default=False)
        If True, perform bootstrap sampling in the leaf nodes.
    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Used when leaf_prediction is 'perceptron'.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import RegressionGenerator
    >>> from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
    >>> import numpy as np
    >>>
    >>> # Setup a data stream
    >>> stream = RegressionGenerator(random_state=1, n_samples=200)
    >>> # Prepare stream for use
    >>>
    >>> # Setup the Hoeffding Adaptive Tree Regressor
    >>> hat_reg = HoeffdingAdaptiveTreeRegressor()
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
    >>>     y_pred[n_samples] = hat_reg.predict(X)[0]
    >>>     hat_reg.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Hoeffding Adaptive Tree regressor mean absolute error: {}'.
    >>>       format(np.mean(np.abs(y_true - y_pred))))
    """

    _ERROR_WIDTH_THRESHOLD = 300
    # ======================================================
    # == Hoeffding Adaptive Tree Regressor implementation ==
    # ======================================================

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
                 nominal_attributes=None,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True,
                 bootstrap_sampling=False,
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
                         nominal_attributes=nominal_attributes,
                         learning_ratio_perceptron=learning_ratio_perceptron,
                         learning_ratio_decay=learning_ratio_decay,
                         learning_ratio_const=learning_ratio_const,
                         leaf_prediction=leaf_prediction,
                         random_state=random_state)
        self.bootstrap_sampling = bootstrap_sampling
        self.alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0

    def _new_learning_node(self, initial_stats=None, parent_node=None,
                           is_active=True):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if initial_stats is None:
            initial_stats = {}

        if is_active:
            return AdaActiveLearningNodeRegressor(initial_stats, parent_node,
                                                  random_state=self.random_state)
        else:
            prediction_option = self.leaf_prediction
            if prediction_option == self._TARGET_MEAN:
                return InactiveLearningNodeMean
            else:
                return InactiveLearningNodePerceptron

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
            self.sum_of_attribute_values += weight * X
            self.sum_of_attribute_squares += weight * X * X
        except ValueError:

            self.sum_of_attribute_values = weight * X
            self.sum_of_attribute_squares = weight * X * X

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        self._tree_root.learn_one(X, y, weight, self, None, -1)

    def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch,
                                  update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, split_parent, parent_branch,
                                                  update_splitter_counts, nodes)
        return nodes

    def _get_votes_for_instance(self, X):
        result = {}
        if self._tree_root is not None:
            if isinstance(self._tree_root, InactiveLeaf):
                found_node = [self._tree_root.filter_instance_to_leaf(X, None, -1)]
            else:
                found_node = self.filter_instance_to_leaves(X, -np.inf, -np.inf, None, -1, False)
            for fn in found_node:
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.get_class_votes(X, self)
                    # add elements to dictionary
                    result = add_dict_values(result, dist, inplace=True)
        return result

    def _new_split_node(self, split_test, class_observations):
        return AdaSplitNodeRegressor(split_test, class_observations, self.random_state)
