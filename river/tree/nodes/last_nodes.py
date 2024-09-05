from __future__ import annotations

from river.tree.utils import BranchFactory
from river.utils.norm import normalize_values_in_dict

from ..splitter.nominal_splitter_classif import NominalSplitterClassif
from ..utils import do_naive_bayes_prediction, round_sig_fig
from .leaf import HTLeaf


class LeafMajorityClassWithDetector(HTLeaf):
    """Leaf that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter,change_detector, split_criterion = None, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.change_detector = change_detector
        self.split_criterion = split_criterion #if None, the change detector will have binary inputs

    @staticmethod
    def new_nominal_splitter():
        return NominalSplitterClassif()
    
    def learn_one(self, x, y, *, w=1, tree=None):
        self.update_stats(y, w)
        if self.is_active():
            if self.split_criterion is None:
                mc_pred = self.prediction(x)
                detector_input = (max(mc_pred, key=mc_pred.get) != y) 
                self.change_detector.update(detector_input)
            else:
                detector_input = self.split_criterion.purity(self.stats)
                self.change_detector.update(detector_input)
            self.update_splitters(x, y, w, tree.nominal_attributes)


    def update_stats(self, y, w):
        try:
            self.stats[y] += w
        except KeyError:
            self.stats[y] = w

    def prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        maj_class = max(self.stats.values())
        # Only perform split attempts when the majority class does not dominate
        # the amount of observed instances
        if maj_class and maj_class / self.total_weight > tree.max_share_to_split:
            return [BranchFactory()]

        return super().best_split_suggestions(criterion, tree)

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self.stats.values())
        if total_seen > 0:
            return total_seen - max(self.stats.values())
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
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2

    def __repr__(self):
        if not self.stats:
            return ""

        text = f"Class {max(self.stats, key=self.stats.get)}:"
        for label, proba in sorted(normalize_values_in_dict(self.stats, inplace=False).items()):
            text += f"\n\tP({label}) = {round_sig_fig(proba)}"

        return text

    def deactivate(self):
        super().deactivate()


class LeafNaiveBayesWithDetector(LeafMajorityClassWithDetector):
    """Leaf that uses Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter,change_detector, split_criterion = None, **kwargs):
        super().__init__(stats, depth, splitter,change_detector,split_criterion,**kwargs)
    
    def learn_one(self, x, y, *, w=1, tree=None):
        self.update_stats(y, w)
        if self.is_active():
            if self.split_criterion is None:
                nb_pred = self.prediction(x)
                detector_input = (max(nb_pred, key=nb_pred.get) == y) 
                self.change_detector.update(detector_input)
            else:
                detector_input = self.split_criterion.purity(self.stats)
                self.change_detector.update(detector_input)
            self.update_splitters(x, y, w, tree.nominal_attributes)

    def prediction(self, x, *, tree=None):
        if self.is_active() and self.total_weight >= tree.nb_threshold:
            return do_naive_bayes_prediction(x, self.stats, self.splitters)
        else:
            return super().prediction(x)

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


class LeafNaiveBayesAdaptiveWithDetector(LeafMajorityClassWithDetector):
    """Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, change_detector,split_criterion = None, **kwargs):
        super().__init__(stats, depth, splitter, change_detector, split_criterion,**kwargs)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_one(self, x, y, *, w=1.0, tree=None):
        """Update the node with the provided instance.

        Parameters
        ----------
        x
            Instance attributes for updating the node.
        y
            Instance class.
        w
            The instance's weight.
        tree
            The Hoeffding Tree to update.

        """
        detector_input_mc = 1
        detector_input_nb = 1
        if self.is_active():
            mc_pred = super().prediction(x)
            # Empty node (assume the majority class will be the best option) or majority
            # class prediction is correct
            if len(self.stats) == 0 or max(mc_pred, key=mc_pred.get) == y:
                self._mc_correct_weight += w
                detector_input_mc = 0
            nb_pred = do_naive_bayes_prediction(x, self.stats, self.splitters)
            if len(nb_pred) > 0 and max(nb_pred, key=nb_pred.get) == y:
                self._nb_correct_weight += w
                detector_input_nb = 0

        self.update_stats(y, w)
        if self.is_active():
            if self.split_criterion is None:
                if self._nb_correct_weight >= self._mc_correct_weight:
                    self.change_detector.update(detector_input_nb)
                else:
                    self.change_detector.update(detector_input_mc)
            else:
                detector_input = self.split_criterion.purity(self.stats)
                self.change_detector.update(detector_input)
            self.update_splitters(x, y, w, tree.nominal_attributes)
        
    

        
    def prediction(self, x, *, tree=None):
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
            return do_naive_bayes_prediction(x, self.stats, self.splitters)
        else:
            return super().prediction(x)

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
