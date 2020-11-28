from river.utils.skmultiflow_utils import normalize_values_in_dict

from .._tree_utils import do_naive_bayes_prediction
from .._attribute_observer import NominalAttributeClassObserver
from .._attribute_observer import NumericAttributeClassObserverBinaryTree
from .._attribute_observer import NumericAttributeClassObserverGaussian
from .._attribute_observer import NumericAttributeClassObserverHistogram

from .base import LearningNode


class LearningNodeMC(LearningNode):
    """Learning node that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params):
        super().__init__(stats, depth, attr_obs, attr_obs_params)

    @staticmethod
    def new_nominal_attribute_observer():
        return NominalAttributeClassObserver()

    @staticmethod
    def new_numeric_attribute_observer(attr_obs, attr_obs_params):
        if attr_obs == "bst":
            return NumericAttributeClassObserverBinaryTree()
        elif attr_obs == "gaussian":
            return NumericAttributeClassObserverGaussian(**attr_obs_params)
        elif attr_obs == "histogram":
            return NumericAttributeClassObserverHistogram(**attr_obs_params)

    def update_stats(self, y, sample_weight):
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

    def leaf_prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
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
    """Learning node that uses Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params):
        super().__init__(stats, depth, attr_obs, attr_obs_params)

    def leaf_prediction(self, x, *, tree=None):
        if self.is_active() and self.total_weight >= tree.nb_threshold:
            return do_naive_bayes_prediction(x, self.stats, self.attribute_observers)
        else:
            return super().leaf_prediction(x)

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
    """Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params):
        super().__init__(stats, depth, attr_obs, attr_obs_params)
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
            mc_pred = super().leaf_prediction(x)
            # Empty node (assume the majority class will be the best option) or majority
            # class prediction is correct
            if len(self.stats) == 0 or max(mc_pred, key=mc_pred.get) == y:
                self._mc_correct_weight += sample_weight

            nb_pred = do_naive_bayes_prediction(x, self.stats, self.attribute_observers)
            if nb_pred is not None and max(nb_pred, key=nb_pred.get) == y:
                self._nb_correct_weight += sample_weight

        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def leaf_prediction(self, x, *, tree=None):
        """Get the probabilities per class for a given instance.

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
            return super().leaf_prediction(x)

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
