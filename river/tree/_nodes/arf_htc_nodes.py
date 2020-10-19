from river.utils.skmultiflow_utils import check_random_state

from .htc_nodes import ActiveLeafClass
from .htc_nodes import LearningNodeMC
from .htc_nodes import LearningNodeNB
from .htc_nodes import LearningNodeNBA


class RandomActiveLeafClass(ActiveLeafClass):
    """Random Active Leaf

    The Random Active Leaf (used in ARF) changes the way in which the nodes
    update the attribute observers (by using subsets of features).
    """
    def update_attribute_observers(self, x, y, sample_weight, tree):
        if len(self.feature_indices) == 0:
            self.feature_indices = self._sample_features(x)

        for idx in self.feature_indices:
            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if tree.nominal_attributes is not None and idx in tree.nominal_attributes:
                    obs = self.new_nominal_attribute_observer()
                else:
                    obs = self.new_numeric_attribute_observer()
                self.attribute_observers[idx] = obs
            obs.update(x[idx], y, sample_weight)

    def _sample_features(self, x):
        selected = self._rng.choice(len(x), size=self.max_features, replace=False)
        features = list(x.keys())

        return [features[s] for s in selected]


class RandomActiveLearningNodeMC(LearningNodeMC, RandomActiveLeafClass):
    """ARF learning node that uses the majority class to provide responses.

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []


class RandomActiveLearningNodeNB(LearningNodeNB, RandomActiveLeafClass):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []


class RandomActiveLearningNodeNBA(LearningNodeNBA, RandomActiveLeafClass):
    """ARF Naive Bayes Adaptive learning node class.

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []
