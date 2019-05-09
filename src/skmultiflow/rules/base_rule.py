from skmultiflow.core import BaseSKMObject


class Rule(BaseSKMObject):
    """ Basic rule class.
    A rule is collection of predicates that build a conjunction (the
    IF part of the rule).

    Typical form of rule:
        * If :math:`Att_{i} > value_{i}` and :math:`Att_{j} = value_{j}` then class_{c}.

    the rule can also track the class distribution and use a drift
    detector to track change in concept.


    parameters
    ----------
    class_distribution: dict (class_value, weight)
        Class observations collected from the instances seen in the rule.
    drift_detector: BaseDriftDetector
        The drift detector used to signal the change in the concept.
    class_idx: int
        The class that rule is describing

    """

    def __init__(self, class_distribution=None, drift_detector=None, class_idx=None):
        """ Rule class constructor"""

        if class_distribution is None:
            self._observed_class_distribution = {}
        else:
            self._observed_class_distribution = class_distribution
        self._drift_detector = drift_detector
        self.predicate_set = []
        self._class_idx = class_idx

    @property
    def drift_detector(self):
        return self._drift_detector

    @drift_detector.setter
    def drift_detector(self, drift_detector):
        self._drift_detector = drift_detector

    @property
    def observed_class_distribution(self):
        return self._observed_class_distribution

    @observed_class_distribution.setter
    def observed_class_distribution(self, dist):
        self._observed_class_distribution = dist

    @property
    def class_idx(self):
        return self._class_idx

    @class_idx.setter
    def class_idx(self, class_idx):
        self._class_idx = class_idx

    def covers_instance(self, X):
        """ Check if the rule covers the instance X.

        parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes to test on the rule.

        returns
        -------
        Boolean
            True if the rule covers the instance else False.

        """
        for predicate in self.predicate_set:
            if not predicate.covers_instance(X):
                return False
        return True

    def restart(self):
        """ Restarts the rule with initial values"""
        self.predicate_set = []
        self.class_idx = None
        self.observed_class_distribution = {}
        if self.drift_detector is not None:
            self.drift_detector.reset()

    def get_rule(self):
        """ Get the rule

        Returns
        -------
        string
            Full description of the rule.
        """
        rule = ""
        for predicate in self.predicate_set:
            rule += " and " + predicate.get_predicate()
        if self.class_idx is not None:
            rule += " | class: " + str(self.class_idx)
        return rule[5:]

    def __str__(self):
        """ Print the rule

        Returns
        -------
        string
            Full description of the rule.
        """
        rule = ""
        for predicate in self.predicate_set:
            rule += " and " + predicate.get_predicate()
        if self.class_idx is not None:
            rule += " | class: " + str(self.class_idx)
        return rule[5:]

    def __eq__(self, other):
        if isinstance(other, Rule):
            if len(other.predicate_set) == len(self.predicate_set):
                for pred, other_pred in zip(self.predicate_set, other.predicate_set):
                    if pred != other_pred:
                        return False
                return True
        return False
