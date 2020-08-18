from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.utils import check_random_state

from ._nodes import RandomActiveLearningNodeMean
from ._nodes import InactiveLearningNodeMean
from ._nodes import RandomActiveLearningNodePerceptron
from ._nodes import InactiveLearningNodePerceptron


class ARFHoeffdingTreeRegressor(HoeffdingTreeRegressor):
    """ ARF Hoeffding Tree regressor.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=2000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=50)
        Number of instances a leaf should observe between split attempts.
    split_confidence: float (default=0.01)
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
    nominal_attributes: list, optional (default: None)
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    learning_ratio_perceptron: float (default: 0.02)
        The learning rate of the perceptron.
    learning_ratio_decay: float (default: 0.001)
        Decay multiplier for the learning rate of the perceptron
    learning_ratio_const: boolean (default: True)
        If False the learning ratio will decay with the number of examples seen
    max_features: int (default: 2)
        Number of randomly selected features to act as split candidates at each attempt.
    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Used when leaf_prediction is 'perceptron' and to randomly pick subsets of
       features at each split.

    This is the base-estimator of the Adaptive Random Forest Regressor ensemble learner (see
    skmultiflow.meta.adaptive_random_forest_regressor). This Hoeffding Tree Regressor includes a
    max_features parameter, which defines the number of randomly selected features to be considered
    at each split.
    """

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_confidence=0.01,
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
                 max_features=2,
                 random_state=None):
        super().__init__(max_byte_size=max_byte_size,
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
                         learning_ratio_const=learning_ratio_const)

        self.max_features = max_features
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

    def _new_learning_node(self, initial_stats=None, parent_node=None,
                           is_active=True):
        """Create a new learning node. The type of learning node depends on the tree
        configuration."""
        if initial_stats is None:
            initial_stats = {}

        # Generate a random seed for the new learning node
        random_state = self._random_state.randint(0, 4294967295, dtype='u8')
        if is_active:
            if self.leaf_prediction == self._TARGET_MEAN:
                return RandomActiveLearningNodeMean(
                    initial_stats, max_features=self.max_features, random_state=random_state)
            elif self.leaf_prediction == self._PERCEPTRON:
                return RandomActiveLearningNodePerceptron(
                    initial_stats, max_features=self.max_features, parent_node=parent_node,
                    random_state=random_state)
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats)
            elif self.leaf_prediction == self._PERCEPTRON:
                return InactiveLearningNodePerceptron(initial_stats, parent_node,
                                                      random_state=random_state)

    def reset(self):
        super().reset()
        self._random_state = check_random_state(self.random_state)

    def new_instance(self):
        return ARFHoeffdingTreeRegressor(max_byte_size=self.max_byte_size,
                                         memory_estimate_period=self.memory_estimate_period,
                                         grace_period=self.grace_period,
                                         split_confidence=self.split_confidence,
                                         tie_threshold=self.tie_threshold,
                                         binary_split=self.binary_split,
                                         stop_mem_management=self.stop_mem_management,
                                         remove_poor_atts=self.remove_poor_atts,
                                         leaf_prediction=self.leaf_prediction,
                                         no_preprune=self.no_preprune,
                                         nominal_attributes=self.nominal_attributes,
                                         learning_ratio_perceptron=self.learning_ratio_perceptron,
                                         learning_ratio_decay=self.learning_ratio_decay,
                                         learning_ratio_const=self.learning_ratio_const,
                                         max_features=self.max_features,
                                         random_state=self.random_state)
