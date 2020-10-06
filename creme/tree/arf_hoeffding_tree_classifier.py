from creme.tree import HoeffdingTreeClassifier
from creme.utils.skmultiflow_utils import check_random_state

from ._nodes import RandomActiveLearningNodeMC
from ._nodes import RandomActiveLearningNodeNB
from ._nodes import RandomActiveLearningNodeNBA
from ._nodes import InactiveLearningNodeMC


class ARFHoeffdingTreeClassifier(HoeffdingTreeClassifier):
    """ Adaptive Random Forest Hoeffding Tree Classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_criterion
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Helinger Distance
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    binary_split
        If True, only allow binary splits.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    max_features
        Number of randomly selected features to act as split candidates at each attempt.
    random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    Notes
    -----
    This is the base-estimator of the Adaptive Random Forest ensemble learner
    (see river.ensemble.AdaptiveRandomForestClassifier). This Hoeffding Tree includes a
    `max_features` parameter, which defines the number of randomly selected features to be
    considered at each split.
    """
    def __init__(self,
                 grace_period: int = 200,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 binary_split: bool = False,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 max_features: int = 2,
                 random_state=None,
                 **kwargs):
        super().__init__(grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.max_features = max_features
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
        if initial_stats is None:
            initial_stats = {}

        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        # Generate a random seed for the new learning node
        random_state = self._random_state.randint(0, 4294967295, dtype='u8')

        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return RandomActiveLearningNodeMC(
                    initial_stats, depth, self.max_features, random_state)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return RandomActiveLearningNodeNB(
                    initial_stats, depth, self.max_features, random_state)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return RandomActiveLearningNodeNBA(
                    initial_stats, depth, self.max_features, random_state)
        else:
            return InactiveLearningNodeMC(initial_stats, depth)

    def reset(self):
        super().reset()
        self._random_state = check_random_state(self.random_state)

    def new_instance(self):
        return self.__class__(self._get_params())
