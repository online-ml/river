import numpy as np

from skmultiflow.utils import check_random_state
from skmultiflow.trees._attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees._attribute_observer import NumericAttributeRegressionObserver
from .htr_nodes import LearningNodeMean, LearningNodePerceptron
from .arf_htc_nodes import RandomActiveLeafClass


class RandomActiveLeafRegressor(RandomActiveLeafClass):
    @staticmethod
    def new_nominal_attribute_observer():
        return NominalAttributeRegressionObserver()

    @staticmethod
    def new_numeric_attribute_observer():
        return NumericAttributeRegressionObserver()

    def manage_memory(self, criterion, last_check_ratio, last_check_sdr, last_check_e):
        """ Trigger Attribute Observers' memory management routines.

        Currently, only `NumericAttributeRegressionObserver` has support to this feature.

        Parameters
        ----------
            criterion: SplitCriterion
                Split criterion
            last_check_ratio: float
                The ratio between the second best candidate's merit and the merit of the best
                split candidate.
            last_check_sdr: float
                The best candidate's split merit.
            last_check_e: float
                Hoeffding bound value calculated in the last split attempt.
        """
        for obs in self.attribute_observers.values():
            if isinstance(obs, NumericAttributeRegressionObserver):
                obs.remove_bad_splits(criterion=criterion, last_check_ratio=last_check_ratio,
                                      last_check_sdr=last_check_sdr, last_check_e=last_check_e,
                                      pre_split_dist=self._stats)


class RandomActiveLearningNodeMean(LearningNodeMean, RandomActiveLeafRegressor):
    """ Learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    max_features: int
        Number of attributes per subset for each node split.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, max_features=2, random_state=None):
        """ ActiveLearningNodeForRegression class constructor. """
        super().__init__(initial_stats)

        self.max_features = max_features
        self.feature_indices = np.array([])
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)


class RandomActiveLearningNodePerceptron(LearningNodePerceptron, RandomActiveLeafRegressor):
    """ Learning Node for regression tasks that always use a linear perceptron
    model to provide responses.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    max_features: int
        Number of attributes per subset for each node split.
    parent_node: RandomLearningNodePerceptron (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats=None, max_features=2, parent_node=None,
                 random_state=None):
        super().__init__(initial_stats, parent_node, random_state)
        self.max_features = max_features
        self.feature_indices = np.array([])
