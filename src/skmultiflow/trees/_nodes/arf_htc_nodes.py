import numpy as np

from skmultiflow.utils import get_dimensions, check_random_state

from .htc_nodes import ActiveLeafClass
from .htc_nodes import LearningNodeMC
from .htc_nodes import LearningNodeNB
from .htc_nodes import LearningNodeNBA


class RandomActiveLeafClass(ActiveLeafClass):
    """ Random Active Leaf

    A Random Active Leaf (used in ARF implementations) just changes the way how the nodes update
    the attribute observers (by using subsets of features).
    """
    def update_attribute_observers(self, X, y, weight, tree):
        if self.feature_indices.size == 0:
            self.feature_indices = self._sample_features(get_dimensions(X)[1])

        for idx in self.feature_indices:
            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if tree.nominal_attributes is not None and idx in tree.nominal_attributes:
                    obs = self.new_nominal_attribute_observer()
                else:
                    obs = self.new_numeric_attribute_observer()
                self.attribute_observers[idx] = obs
            obs.update(X[idx], y, weight)

    def _sample_features(self, n_features):
        return self._random_state.choice(
            n_features, size=self.max_features, replace=False
        )


class RandomActiveLearningNodeMC(LearningNodeMC, RandomActiveLeafClass):
    """ARF learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, max_features=2, random_state=None):
        """ RandomLearningNodeClassification class constructor. """
        super().__init__(initial_stats)
        self.max_features = max_features
        self.feature_indices = np.array([])
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)


class RandomActiveLearningNodeNB(LearningNodeNB, RandomActiveLeafClass):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats=None, max_features=2, random_state=None):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_stats)
        self.max_features = max_features
        self.feature_indices = np.array([])
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)


class RandomActiveLearningNodeNBA(LearningNodeNBA, RandomActiveLeafClass):
    """Naive Bayes Adaptive learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, max_features=2, random_state=None):
        """LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_stats)
        self.max_features = max_features
        self.feature_indices = np.array([])
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
