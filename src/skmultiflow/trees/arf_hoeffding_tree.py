from skmultiflow.utils import check_random_state
from skmultiflow.trees.hoeffding_tree import HoeffdingTree, MAJORITY_CLASS, \
    NAIVE_BAYES
from skmultiflow.trees.nodes import RandomLearningNodeClassification
from skmultiflow.trees.nodes import RandomLearningNodeNB
from skmultiflow.trees.nodes import RandomLearningNodeNBAdaptive


class ARFHoeffdingTree(HoeffdingTree):
    """ Adaptive Random Forest Hoeffding Tree.

    Parameters
    ----------
    max_byte_size: int, optional (default=33554432)
        Maximum memory consumed by the tree.

    memory_estimate_period: int, optional (default=2000000)
        Number of instances between memory consumption checks.

    grace_period: int, optional (default=50)
        Number of instances a leaf should observe between split attempts.

    split_criterion: string, optional (default='info_gain')
        Split criterion to use.

        - 'gini' - Gini
        - 'info_gain' - Information Gain

    split_confidence: float, optional (default=0.01)
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.

    tie_threshold: float, optional (default=0.05)
        Threshold below which a split will be forced to break ties.

    binary_split: bool, optional (default=False)
        If True, only allow binary splits.

    stop_mem_management: bool, optional (default=False)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: bool, optional (default=False)
        If True, disable poor attributes.

    no_preprune: bool, optional (default=False)
        If True, disable pre-pruning.

    leaf_prediction: string, optional (default='nba')
        Prediction mechanism used at leafs.

        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive

    nb_threshold: int, optional (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes
        are numerical.

    random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.

    Notes
    -----
    This is the base model for the Adaptive Random Forest ensemble learner
    (See skmultiflow.classification.meta.adaptive_random_forests).
    This Hoeffding Tree includes a max_features parameter, which defines the
    number of randomly selected features to
    be considered at each split.
    """
    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.01,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 max_features=2,
                 random_state=None):
        """ARFHoeffdingTree class constructor."""
        super().__init__(max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes)
        self.max_features = max_features
        self.remove_poor_attributes = False
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the
        tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        # MAJORITY CLASS
        if self._leaf_prediction == MAJORITY_CLASS:
            return RandomLearningNodeClassification(
                initial_class_observations, self.max_features,
                random_state=self._random_state
            )
        # NAIVE BAYES
        elif self._leaf_prediction == NAIVE_BAYES:
            return RandomLearningNodeNB(
                initial_class_observations, self.max_features,
                random_state=self._random_state
            )
        # NAIVE BAYES ADAPTIVE
        else:
            return RandomLearningNodeNBAdaptive(
                initial_class_observations, self.max_features,
                random_state=self._random_state
            )

    @staticmethod
    def is_randomizable():
        return True

    def reset(self):
        super().reset()
        self._random_state = check_random_state(self.random_state)

    def new_instance(self):
        return ARFHoeffdingTree(max_byte_size=self.max_byte_size,
                                memory_estimate_period=self.memory_estimate_period,
                                grace_period=self.grace_period,
                                split_criterion=self.split_criterion,
                                split_confidence=self.split_confidence,
                                tie_threshold=self.tie_threshold,
                                binary_split=self.binary_split,
                                stop_mem_management=self.stop_mem_management,
                                remove_poor_atts=self.remove_poor_atts,
                                no_preprune=self.no_preprune,
                                leaf_prediction=self.leaf_prediction,
                                nb_threshold=self.nb_threshold,
                                nominal_attributes=self.nominal_attributes,
                                max_features=self.max_features,
                                random_state=self._random_state)
