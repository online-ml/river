from river.tree import HoeffdingTreeClassifier
from river.utils.skmultiflow_utils import check_random_state

from ._nodes import RandomLearningNodeMC
from ._nodes import RandomLearningNodeNB
from ._nodes import RandomLearningNodeNBA


class ARFHoeffdingTreeClassifier(HoeffdingTreeClassifier):
    """ Adaptive Random Forest Hoeffding Tree Classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Hellinger Distance
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leaves.
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
    seed: int, RandomState instance or None, optional (default=None)
            If int, seed is the seed used by the random number generator;
            If RandomState instance, seed is the random number generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.
    **kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

    Notes
    -----
    This is the base-estimator of the Adaptive Random Forest ensemble learner
    (see river.ensemble.AdaptiveRandomForestClassifier). This Hoeffding Tree includes a
    `max_features` parameter, which defines the number of randomly selected features to be
    considered at each split.
    """
    def __init__(self,
                 grace_period: int = 200,
                 max_depth: int = None,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 max_features: int = 2,
                 seed=None,
                 **kwargs):
        super().__init__(grace_period=grace_period,
                         max_depth=max_depth,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)

    def _new_learning_node(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        # Generate a random seed for the new learning node
        seed = self._rng.randint(0, 4294967295, dtype='u8')

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return RandomLearningNodeMC(initial_stats, depth, self.max_features, seed)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return RandomLearningNodeNB(initial_stats, depth, self.max_features, seed)
        else:  # NAIVE BAYES ADAPTIVE (default)
            return RandomLearningNodeNBA(initial_stats, depth, self.max_features, seed)

    def new_instance(self):
        return self.__class__(**self._get_params())
