from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.attribute_observer import AttributeClassObserverNull
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nodes import LearningNode


class ActiveLearningNode(LearningNode):
    """ Learning node that supports growth.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ ActiveLearningNode class constructor. """
        super().__init__(initial_class_observations)
        self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
        self._attribute_observers = {}

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTree
            Hoeffding Tree to update.

        """
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
            self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

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

    def get_weight_seen(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self._observed_class_distribution.values())

    def get_weight_seen_at_last_split_evaluation(self):
        """ Retrieve the weight seen at last split evaluation.

        Returns
        -------
        float
            Weight seen at last split evaluation.

        """
        return self._weight_seen_at_last_split_evaluation

    def set_weight_seen_at_last_split_evaluation(self, weight):
        """ Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight: float
            Weight seen at last split evaluation.

        """
        self._weight_seen_at_last_split_evaluation = weight

    def get_best_split_suggestions(self, criterion, ht):
        """ Find possible split candidates.

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.
        ht: HoeffdingTree
            Hoeffding Tree.

        Returns
        -------
        list
            Split candidates.

        """
        best_suggestions = []
        pre_split_dist = self._observed_class_distribution
        if not ht.no_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        for i, obs in self._attribute_observers.items():
            best_suggestion = obs.get_best_evaluated_split_suggestion(
                criterion, pre_split_dist, i, ht.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, att_idx):
        """ Disable an attribute observer.

        Parameters
        ----------
        att_idx: int
            Attribute index.

        """
        if att_idx in self._attribute_observers:
            self._attribute_observers[att_idx] = AttributeClassObserverNull()

    def get_attribute_observers(self):
        """ Get attribute observers at this node.

        Returns
        -------
        dict (attribute id, attribute observer object)
            Attribute observers of this node.

        """
        return self._attribute_observers

    def set_attribute_observers(self, attribute_observers):
        """ set attribute observers.

        Parameters
        ----------
        attribute_observers: dict (attribute id, attribute observer object)
            new attribute observers.

        """
        self._attribute_observers = attribute_observers
