from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.nodes import ActiveLearningNode

NAIVE_BAYES_ADAPTIVE = 'nba'


class LCActiveLearningNode(ActiveLearningNode):
    """ Active Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_class_observations):
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):

        if not(ht.leaf_prediction == NAIVE_BAYES_ADAPTIVE):
            y = ''.join(str(e) for e in y)
            y = int(y, 2)

        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)
