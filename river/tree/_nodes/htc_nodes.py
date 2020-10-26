from river.tree._tree_utils import do_naive_bayes_prediction
from river.tree._attribute_observer import NominalAttributeClassObserver
from river.tree._attribute_observer import NumericAttributeClassObserverGaussian

from .base import LearningNode


class LearningNodeMC(LearningNode):
    """Learning node that always predicts the majority class."""
    @staticmethod
    def new_nominal_attribute_observer(**kwargs):
        return NominalAttributeClassObserver(**kwargs)

    @staticmethod
    def new_numeric_attribute_observer(**kwargs):
        return NumericAttributeClassObserverGaussian(**kwargs)

    def update_stats(self, y, sample_weight):
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

    def predict_one(self, x, *, tree=None):
        return self.stats

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self._stats.values())
        if total_seen > 0:
            return total_seen - max(self._stats.values())
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is less than 2, False otherwise.
        """
        count = 0
        for weight in self._stats.values():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2


class LearningNodeNB(LearningNodeMC):
    """Learning node that uses Naive Bayes models."""
    def predict_one(self, x, *, tree=None):
        if self.is_active() and self.total_weight >= tree.nb_threshold:
            return do_naive_bayes_prediction(x, self.stats, self.attribute_observers)
        else:
            return self.stats

    def disable_attribute(self, att_index):
        """Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index
            Attribute index.
        """
        pass


class LearningNodeNBA(LearningNodeMC):
    """Learning node that uses Adaptive Naive Bayes models."""
    def __init__(self, initial_stats, depth):
        super().__init__(initial_stats, depth)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        """Update the node with the provided instance.

        Parameters
        ----------
        x
            Instance attributes for updating the node.
        y
            Instance class.
        sample_weight
            The instance's weight.
        tree
            The Hoeffding Tree to update.

        """
        if self.is_active():
            if len(self.stats) == 0:
                # Empty node, assume the majority class will be the best option
                self._mc_correct_weight += sample_weight
            elif max(self.stats, key=self.stats.get) == y:  # Majority class
                self._mc_correct_weight += sample_weight

            nb_prediction = do_naive_bayes_prediction(x, self.stats, self.attribute_observers)
            if nb_prediction is not None and max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += sample_weight

        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def predict_one(self, x, *, tree=None):
        """Get the votes per class for a given instance.

        Parameters
        ----------
        x
            Instance attributes.
        tree
            Hoeffding Tree.

        Returns
        -------
        Class votes for the given instance.

        """
        if self.is_active() and self._nb_correct_weight >= self._mc_correct_weight:
            return do_naive_bayes_prediction(x, self.stats, self.attribute_observers)
        else:
            return self.stats

    def disable_attribute(self, att_index):
        """Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index
            Attribute index.
        """
        pass
