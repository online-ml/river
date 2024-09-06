from __future__ import annotations

from ..utils import do_naive_bayes_prediction
from .htc_nodes import LeafMajorityClass


class LeafMajorityClassWithDetector(LeafMajorityClass):
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
    change_detector
        Change detector that monitors the leaf error rate or class distribution and
        determines when the leaf will split.
    split_criterion
        Split criterion used in the tree for updating the change detector if it
        monitors the class distribution.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, change_detector, split_criterion=None, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.change_detector = change_detector
        # change this in future PR's by acessing the tree parameter in the leaf
        self.split_criterion = (
            split_criterion  # if None, the change detector will have binary inputs
        )

    def learn_one(self, x, y, *, w=1, tree=None):
        self.update_stats(y, w)
        if self.is_active():
            if self.split_criterion is None:
                mc_pred = self.prediction(x)
                detector_input = max(mc_pred, key=mc_pred.get) != y
                self.change_detector.update(detector_input)
            else:
                detector_input = self.split_criterion.current_merit(self.stats)
                self.change_detector.update(detector_input)
            self.update_splitters(x, y, w, tree.nominal_attributes)


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
    change_detector
        Change detector that monitors the leaf error rate or class distribution and
        determines when the leaf will split.
    split_criterion
        Split criterion used in the tree for updating the change detector if it
        monitors the class distribution.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, change_detector, split_criterion=None, **kwargs):
        super().__init__(stats, depth, splitter, change_detector, split_criterion, **kwargs)

    def learn_one(self, x, y, *, w=1, tree=None):
        self.update_stats(y, w)
        if self.is_active():
            if self.split_criterion is None:
                nb_pred = self.prediction(x)
                detector_input = max(nb_pred, key=nb_pred.get) == y
                self.change_detector.update(detector_input)
            else:
                detector_input = self.split_criterion.current_merit(self.stats)
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
    change_detector
        Change detector that monitors the leaf error rate or class distribution and
        determines when the leaf will split.
    split_criterion
        Split criterion used in the tree for updating the change detector if it
        monitors the class distribution.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, change_detector, split_criterion=None, **kwargs):
        super().__init__(stats, depth, splitter, change_detector, split_criterion, **kwargs)
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
            The tree to update.

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
                detector_input = self.split_criterion.current_merit(self.stats)
                self.change_detector.update(detector_input)
            self.update_splitters(x, y, w, tree.nominal_attributes)

    def prediction(self, x, *, tree=None):
        """Get the probabilities per class for a given instance.

        Parameters
        ----------
        x
            Instance attributes.
        tree
            LAST Tree.

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
