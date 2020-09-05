from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees._attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees._attribute_observer import NumericAttributeClassObserverGaussian

from .base import LearningNode, ActiveLeaf, InactiveLeaf


class ActiveLeafClass(ActiveLeaf):
    @staticmethod
    def new_nominal_attribute_observer():
        return NominalAttributeClassObserver()

    @staticmethod
    def new_numeric_attribute_observer():
        return NumericAttributeClassObserverGaussian()


class LearningNodeMC(LearningNode):
    def update_stats(self, y, weight):
        try:
            self.stats[y] += weight
        except KeyError:
            self.stats[y] = weight
            self.stats = dict(sorted(self.stats.items()))

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        # Enforce y is an integer
        super().learn_one(X, int(y), weight=weight, tree=tree)

    def predict_one(self, X, *, tree=None):
        return self.stats

    @property
    def total_weight(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self.stats.values())

    def observed_class_distribution_is_pure(self):
        """ Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is less than 2, False otherwise.

        """
        count = 0
        for _, weight in self._stats.items():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2


class LearningNodeNB(LearningNodeMC):
    def predict_one(self, X, *, tree=None):
        if self.total_weight >= tree.nb_threshold:
            return do_naive_bayes_prediction(X, self.stats, self.attribute_observers)
        else:
            return self.stats


class LearningNodeNBA(LearningNodeMC):
    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            The instance's weight.
        tree: HoeffdingTreeClassifier
            The Hoeffding Tree to update.

        """
        if self.stats == {}:
            # All classes equal, default to class 0
            if y == 0:
                self._mc_correct_weight += weight
        elif max(self.stats, key=self.stats.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(X, self.stats, self.attribute_observers)
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight

        super().learn_one(X, y, weight=weight, tree=tree)

    def predict_one(self, X, *, tree=None):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        tree: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        if self._mc_correct_weight > self._nb_correct_weight:
            return self.stats
        return do_naive_bayes_prediction(X, self.stats, self.attribute_observers)


class ActiveLearningNodeMC(LearningNodeMC, ActiveLeafClass):
    """ Learning node that supports growth.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)


class InactiveLearningNodeMC(LearningNodeMC, InactiveLeaf):
    """ Inactive learning node that does not grow.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)


class ActiveLearningNodeNB(LearningNodeNB, ActiveLeafClass):
    """ Learning node that uses Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)

    def disable_attribute(self, att_index):
        """ Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index: int
            Attribute index.
        """
        pass


class ActiveLearningNodeNBA(LearningNodeNBA, ActiveLeafClass):
    """ Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)

    def disable_attribute(self, att_index):
        """ Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index: int
            Attribute index.
        """
        pass
