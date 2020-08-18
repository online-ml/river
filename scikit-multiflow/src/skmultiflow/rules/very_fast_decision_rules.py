import copy
from operator import attrgetter, itemgetter
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.rules.base_predicate import Predicate
from skmultiflow.rules.base_rule import Rule
from skmultiflow.rules.foil_gain_rule_criterion import FoilGainExpandCriterion
from skmultiflow.rules.hellinger_distance_criterion import HellingerDistanceCriterion
from skmultiflow.rules.info_gain_rule_criterion import InfoGainExpandCriterion
from skmultiflow.rules.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.rules.numeric_attribute_class_observer import GaussianNumericAttributeClassObserver
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.utils import get_dimensions, normalize_values_in_dict, \
    calculate_object_size
from skmultiflow.trees._attribute_observer import AttributeObserverNull


_FIRSTHIT = 'first_hit'
_WEIGHTEDMAX = 'weighted_max'
_WEIGHTEDSUM = 'weighted_sum'
_INFOGAIN = 'info_gain'
_FOILGAIN = 'foil_gain'
_HELLINGER = 'hellinger'
_EDDM = 'eddm'
_ADWIN = 'adwin'
_DDM = 'ddm'

import warnings


def VFDR(expand_confidence=0.0000001, ordered_rules=True, grace_period=200, tie_threshold=0.05,
         rule_prediction='first_hit', nominal_attributes=None, max_rules=1000, nb_threshold=0, nb_prediction=True,
         drift_detector=None, expand_criterion='info_gain', remove_poor_atts=False, min_weight=100):  # pragma: no cover
    warnings.warn("'VFDR' has been renamed to 'VeryFastDecisionRulesClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return VeryFastDecisionRulesClassifier(expand_confidence=expand_confidence,
                                           ordered_rules=ordered_rules,
                                           grace_period=grace_period,
                                           tie_threshold=tie_threshold,
                                           rule_prediction=rule_prediction,
                                           nominal_attributes=nominal_attributes,
                                           max_rules=max_rules,
                                           nb_threshold=nb_threshold,
                                           nb_prediction=nb_prediction,
                                           drift_detector=drift_detector,
                                           expand_criterion=expand_criterion,
                                           remove_poor_atts=remove_poor_atts,
                                           min_weight=min_weight)


class VeryFastDecisionRulesClassifier(BaseSKMObject, ClassifierMixin):
    """ Very Fast Decision Rules classifier.

    The Very Fast Decision Rules (VFDR) [1]_ is an incremental rule learning classifier.
    The learning process of VFDR is similar to that of Hoeffding Tree, but instead of a tree
    it uses a collection of rules. The core of VFDR is its rules that aim to create a highly
    interpretable classifier thanks to their nature. Each rule is a conjunction of conditions
    based on attribute values and the structure for keeping sufficient statistics. The sufficient
    statistics will determine the class predicted by the rule

    IF :math:`att_1 < 1` and :math:`att_2 = 0` THEN class 0.

    The Adaptive Very Fast Decision Rules (AVFDR) is an extension of VFDR capable to adapt to
    evolving data streams. To adapt with the concept every rule is equipped with a drift detector
    to monitor it's performance. If a change is detected then the rule is removed and a new one
    will be learned.

    Parameters
    ----------
    expand_confidence: float (default=0.0000001)
        | Allowed error in split decision, a value closer to 0 takes longer to decide.
    ordered_rules: Bool (default=True)
        | Whether the created rule set is ordered or not. An ordered set only expands the first
          rule that fires while the unordered set expands all the rules that fire.
    grace_period: int (default=200)
        | Number of instances a leaf should observe between split attempts.
    tie_threshold: float (default=0.05)
        | Threshold below which a split will be forced to break ties.
    rule_prediction: string (default='first_hit')
        | How the class votes are retrieved for prediction. Since more than one rule can fire
          statistics can be gathered in three ways:

        - 'first_hit' - Uses the votes of the first rule that fires.
        - 'weighted_max' - Uses the votes of the rules with highest weight that fires.
        - 'weighted_sum' - Uses the weighted sum of votes of all the rules that fire.

    nominal_attributes: list, optional
        | List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    max_rules: int (default=20)
        | Maximum number of rules the model can have.
    nb_threshold: int (default=0)
        | Number of instances a leaf should observe before allowing Naive Bayes.
    nb_prediction: Bool (default=True)
        | Use Naive Bayes as prediction strategy in the leafs, else majority class is uses.
    drift_detector: BaseDriftDetector (Default=None)
        | The drift detector to use in rules. If None detection will be ignored.
        | If set, the estimator is effectively the Adaptive Very Fast Decision Rules classifier.
        | Supported detectors: ADWIN, DDM and EDDM.
    expand_criterion: SplitCriterion (Default='info_gain')
        | Expand criterion to use:

        - 'info_gain' - Information Gain
        - 'hellinger' - Hellinger Distance
        - 'foil_gain' - Foil Gain

    remove_poor_atts: boolean (default=False)
        | If True, disable poor attributes.

    Examples
    --------
    >>> from skmultiflow.rules import VeryFastDecisionRulesClassifier
    >>> from skmultiflow.data import AGRAWALGenerator
    >>> # Setup the stream
    >>> stream = AGRAWALGenerator()
    >>> X, y = stream.next_sample(20000)
    >>> # Setup the learner
    >>> learner = VeryFastDecisionRulesClassifier()
    >>> # Train
    >>> learner.partial_fit(X, y)
    >>> # Print rules
    >>> print(learner.get_model_description())
        Rule 0 :Att (2) <= 36.360| class :0  {0: 13867.746092302219}
        Rule 1 :Att (2) > 60.450| class :0  {0: 16174.452447424192}
        Rule 2 :Att (2) <= 39.090| class :0  {0: 2374.5506811568403}
        Rule 3 :Att (2) <= 58.180| class :1  {1: 15082.368305403843}
        Rule 4 :Att (2) <= 59.090| class :1  {1: 767.0}
        Default Rule :| class :0  {0: 837.0}
    >>> # Predict
    >>> X, y = stream.next_sample(100)
    >>> learner.predict(X)
        [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 1 1 1 1 1 0 1 0 0 0 0
         0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0
         1 1 0 1 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0 0 1]

    Notes
    -----
    When using an ordered set, the rules need to be read in order as each rule contain a hidden
    conjunction which is the opposite of all the rules before it. Also only one rule can fire which
    restricts the prediction to first hit.

    References
    ----------
    .. [1] Petr Kosina and João Gama. 2015. Very fast decision rules for classification in
       data streams. Data Min. Knowl. Discov. 29, 1 (January 2015), 168-202.
       DOI=http://dx.doi.org/10.1007/s10618-013-0340-z
    """

    class Rule(Rule):
        """ Rule class

        A rule is collection of predicates(conditions) that make up the conjunction (the IF part of the rule).
        The conditions are in the form of: :math:`Att_{idx} > value`,  :math:`Att_{idx} <= value` and
        :math:`Att_{idx} = value`.

        The rule can also track the class distribution and use a drift detector to track change in concept.

        Parameters
        ----------
        class_distribution: dict (class_value, weight)
            Class observations collected from the instances seen in the rule.
        drift_detector: BaseDriftDetector (Default=None)
            The drift detector used to signal the change in the concept.

        """

        def __init__(self, class_distribution, drift_detector, class_idx):
            """ Rule class constructor"""
            super().__init__(class_distribution=class_distribution, drift_detector=drift_detector, class_idx=class_idx)
            self._weight_seen_at_last_expand = self.get_weight_seen()
            self._attribute_observers = {}

        @property
        def weight_seen_at_last_expand(self):
            """Retrieve the weight seen at last expand evaluation.

            Returns
            -------
            float
                Weight seen at last expand evaluation.

            """
            return self._weight_seen_at_last_expand

        @weight_seen_at_last_expand.setter
        def weight_seen_at_last_expand(self, weight):
            """Set the weight seen at last expand evaluation.

            Parameters
            ----------
            weight: float
                Weight seen at last expand evaluation.

            """
            self._weight_seen_at_last_expand = weight

        def learn_from_instance(self, X, y, weight, avfdr):
            """Update the rule with the provided instance.
            The model class distribution of the model and each attribute are updated. The one for the model is used
            for prediction and the distributions for the attributes are used for learning
            Gaussian estimators are used to track distribution of numeric attributes and dict with class count
            for nominal and the model distributions.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            avfdr: AVFDR
                Very Fast Decision Rule model.

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
                    if i in avfdr.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.update(X[i], int(y), weight)

        def get_weight_seen(self):
            """Calculate the total weight seen by the node.

            Returns
            -------
            float
                Total weight seen.

            """
            return sum(self.observed_class_distribution.values()) if self.observed_class_distribution != {} else 0

        def get_best_expand_suggestion(self, criterion, class_idx):
            """Find possible expand candidates.

            Parameters
            ----------
            criterion: Splitriterion
                The criterion used to chose the best expanding suggestion.
            class_idx: int or None
                if foil gain is used as a criterion class_idx is the class rule tries to learn.

            Returns
            -------
            list
                expand candidates.

            """
            best_suggestions = []
            pre_expand_dist = self.observed_class_distribution
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_expand_dist, i, class_idx)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)

            return best_suggestions

        def get_class_votes(self, X, vfdr):
            """Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.

            vfdr: AVFDR
                Very Fast Decision Rules.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self.get_weight_seen() >= vfdr.nb_threshold and vfdr.nb_prediction:
                return do_naive_bayes_prediction(X, self.observed_class_distribution, self._attribute_observers)
            else:
                return self.observed_class_distribution

        def disable_attribute(self, att_idx):
            """Disable an attribute observer.

            Parameters
            ----------
            att_idx: int
                Attribute index.

            """
            if att_idx in self._attribute_observers:
                self._attribute_observers[att_idx] = AttributeObserverNull()

        def restart(self):
            """ Restarts the rule with initial values"""
            super().restart()
            self._attribute_observers = {}
            self.weight_seen_at_last_expand = self.get_weight_seen()

        def predict(self, y):
            """
            Provides information about the classification of the rule for the
            drift detector in order to follow it's performance.

            Parameters
            ----------
            y: int
                The true label

            Returns
            -------
            int
                1 if the prediction is correct else 0

            """
            votes = self.observed_class_distribution
            if votes == {}:
                prediction = 0
            else:
                prediction = max(votes.items(), key=itemgetter(1))[0]
            return 1 if prediction == y else 0

    def new_rule(self, class_distribution=None, drift_detector=None, class_idx=None):
        """ Create a new rule.

        Parameters
        ----------
        class_distribution: dict (class_value, weight)
            Class observations collected from the instances seen in the rule.
        drift_detector: BaseDriftDetector
            The drift detector used to signal the change in the concept.
        class_idx: int or None
            The class the rule is describing.

        Returns
        -------
        Rule:
            The created rule.

        """
        return self.Rule(class_distribution, drift_detector, class_idx)

    def get_votes_for_instance(self, X):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        if self.rule_prediction == _FIRSTHIT:
            return self.first_hit(X)
        elif self.rule_prediction == _WEIGHTEDMAX:
            return self.weighted_max(X)
        elif self.rule_prediction == _WEIGHTEDSUM:
            return self.weighted_sum(X)

    def first_hit(self, X):
        """ Get class votes from the first rule that fires.

        parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            The class distribution of the fired rule.

        """

        for rule in self.rule_set:
            if rule.covers_instance(X):
                votes = rule.get_class_votes(X, self).copy()
                return votes
        return self.default_rule.get_class_votes(X, self)

    def weighted_max(self, X):
        """ Get class votes from the rule with highest vote weight.

        parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            the class distribution from the rule with highest weight.

        """

        highest = 0
        final_votes = self.default_rule.get_class_votes(X, self)
        for rule in self.rule_set:
            if rule.covers_instance(X):
                votes = copy.deepcopy(rule.get_class_votes(X, self))
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                for v in votes.values():
                    if v >= highest:
                        highest = v
                        final_votes = votes

        return final_votes

    def weighted_sum(self, X):
        """ Get class votes from the sum of rules that fires.
         The rules are weighted.

        parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)
            The class distribution from the sum of the fired rules.

        """
        final_votes = {}
        fired_rule = False
        for rule in self.rule_set:
            if rule.covers_instance(X):
                fired_rule = True
                votes = copy.deepcopy(rule.get_class_votes(X, self))
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                final_votes = {k: final_votes.get(k, 0) + votes.get(k, 0) for k in set(final_votes) | set(votes)}
                if sum(final_votes.values()) != 0:
                    normalize_values_in_dict(final_votes)

        return final_votes if fired_rule else self.default_rule.get_class_votes(X, self)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally trains the model.

        Train samples (instances) are composed of X attributes and their corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the rule_set is empty, update the default_rule and if enough statistics are collected try to create rule.
        * If rules exist in the rule_set, check if they cover the instance. The statistics of the ones that fire are
          updated using the instance.
        * If enough statistics are collected if a rule then attempt to expand it.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        self

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], sample_weight[i])

        return self

    def _partial_fit(self, X, y, weight):
        """Trains the model on sample X and corresponding target y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        rule_fired = False
        if self.rule_set:
            for i, rule in enumerate(self.rule_set):
                if rule.covers_instance(X):
                    rule_fired = True
                    if self.drift_detector is not None:
                        prediction = rule.predict(y)
                        rule.drift_detector.add_element(prediction)
                        if rule.drift_detector.detected_change() and rule.get_weight_seen() > self.min_weight:
                            self.rule_set.pop(i)
                            continue
                    rule.learn_from_instance(X, y, weight, self)
                    if rule.get_weight_seen() - rule.weight_seen_at_last_expand >= self.grace_period:
                        self._expand_rule(rule)
                        rule.weight_seen_at_last_expand = rule.get_weight_seen()
                    if self.ordered_rules:
                        break
        if not rule_fired:
            self.default_rule.learn_from_instance(X, y, weight, self)
            if self.max_rules > len(self.rule_set):
                if self.default_rule.get_weight_seen() - self.default_rule.weight_seen_at_last_expand >= \
                        self.grace_period:
                    self._create_rule()

    def _create_rule(self):
        """ Create a new rule from the default rule.

        If the default rule has enough statistics, possible expanding candidates are checked.
        If the best candidate verifies the Hoeffding bound, a new rule is created if a one predicate.
        The rule statistics are passed down to the new rule and the default rule is reset.

        """
        if len(self.default_rule.observed_class_distribution) >= 2:
            if self.expand_criterion in [_INFOGAIN, _HELLINGER]:
                if self.expand_criterion == _HELLINGER:
                    expand_criterion = HellingerDistanceCriterion()
                else:
                    expand_criterion = InfoGainExpandCriterion()
                should_expand = False
                best_expand_suggestions = self.default_rule.get_best_expand_suggestion(expand_criterion, None)
                best_expand_suggestions.sort(key=attrgetter('merit'))

                if len(best_expand_suggestions) < 2:
                    should_expand = len(best_expand_suggestions) > 0
                else:
                    hoeffding_bound = self.compute_hoeffding_bound(expand_criterion.get_range_of_merit(
                        self.default_rule.observed_class_distribution), self.expand_confidence,
                        self.default_rule.get_weight_seen())
                    best_suggestion = best_expand_suggestions[-1]
                    second_best_suggestion = best_expand_suggestions[-2]

                    if ((best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound) or \
                            (hoeffding_bound < self.tie_threshold):
                        should_expand = True

                if should_expand:
                    best_suggestion = best_expand_suggestions[-1]
                    new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator, best_suggestion.att_val)
                    self.rule_set.append(self.new_rule(None, copy.deepcopy(self.drift_detector), None))
                    self.rule_set[-1].predicate_set.append(new_pred)
                    self.default_rule.restart()
                    if new_pred.operator in ["=", "<="]:
                        self.rule_set[-1].observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(0).copy()
                        self.default_rule.observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(1).copy()
                    else:
                        self.rule_set[-1].observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(1).copy()
                        self.default_rule.observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(0).copy()
                else:
                    self.default_rule.weight_seen_at_last_expand = self.default_rule.get_weight_seen()
            elif self.expand_criterion == _FOILGAIN:
                expand_criterion = FoilGainExpandCriterion()
                should_expand = False
                for class_idx in self.default_rule.observed_class_distribution.keys():
                    best_expand_suggestions = self.default_rule.get_best_expand_suggestion(expand_criterion, class_idx)
                    best_expand_suggestions.sort(key=attrgetter('merit'))
                    if len(best_expand_suggestions) < 2:
                        should_expand = len(best_expand_suggestions) > 0
                    else:
                        hoeffding_bound = self.compute_hoeffding_bound(expand_criterion.get_range_of_merit(
                            self.default_rule.observed_class_distribution), self.expand_confidence,
                            self.default_rule.get_weight_seen())
                        best_suggestion = best_expand_suggestions[-1]
                        second_best_suggestion = best_expand_suggestions[-2]
                        if ((best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound) or (
                                hoeffding_bound < self.tie_threshold):
                            should_expand = True
                    if should_expand:
                        best_suggestion = best_expand_suggestions[-1]
                        new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator, best_suggestion.att_val)
                        self.rule_set.append(self.new_rule(None, copy.deepcopy(self.drift_detector), class_idx))
                        self.rule_set[-1].predicate_set.append(new_pred)
                        if new_pred.operator in ["=", "<="]:
                            self.rule_set[-1].observed_class_distribution = best_suggestion. \
                                resulting_stats_from_split(0).copy()
                        else:
                            self.rule_set[-1].observed_class_distribution = best_suggestion. \
                                resulting_stats_from_split(1).copy()
                if should_expand:
                    self.default_rule.restart()
                else:
                    self.default_rule.weight_seen_at_last_expand = self.default_rule.get_weight_seen()
        else:
            self.default_rule.weight_seen_at_last_expand = self.default_rule.get_weight_seen()

    def _expand_rule(self, rule):
        """
        If the rule has enough statistics, possible expanding candidates are checked. If the best
        candidate verifies the Hoeffding bound, a new predicate is add to the  rule.
        The rule statistics are update to fit the new description.

        """

        if len(rule.observed_class_distribution) >= 2:
            class_idx = None
            if self.expand_criterion == _HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            elif self.expand_criterion == _INFOGAIN:
                split_criterion = InfoGainExpandCriterion()
            else:
                split_criterion = FoilGainExpandCriterion()
                class_idx = rule.class_idx
            should_expand = False
            best_expand_suggestions = rule.get_best_expand_suggestion(split_criterion, class_idx)
            best_expand_suggestions.sort(key=attrgetter('merit'))

            if len(best_expand_suggestions) < 2:
                should_expand = len(best_expand_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    rule.observed_class_distribution), self.expand_confidence, rule.get_weight_seen())
                best_suggestion = best_expand_suggestions[-1]
                second_best_suggestion = best_expand_suggestions[-2]
                if ((best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound) or \
                        (hoeffding_bound < self.tie_threshold):
                    should_expand = True

                if self.remove_poor_atts is not None and self.remove_poor_atts:
                    poor_atts = set()
                    # Scan 1 - add any poor attribute to set
                    for i in range(len(best_expand_suggestions)):
                        if best_expand_suggestions[i] is not None:
                            split_atts = [best_expand_suggestions[i].att_idx]
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_expand_suggestions[i].merit > hoeffding_bound:
                                    poor_atts.add(int(split_atts[0]))
                    # Scan 2 - remove good attributes from set
                    for i in range(len(best_expand_suggestions)):
                        if best_expand_suggestions[i] is not None:
                            split_atts = [best_expand_suggestions[i].att_idx]
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_expand_suggestions[i].merit < hoeffding_bound:
                                    try:
                                        poor_atts.remove(int(split_atts[0]))
                                    except KeyError:
                                        pass
                    for poor_att in poor_atts:
                        rule.disable_attribute(poor_att)

            if should_expand:
                best_suggestion = best_expand_suggestions[-1]
                new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator, best_suggestion.att_val)
                add_pred = True
                for pred in rule.predicate_set:
                    if (pred.operator == new_pred.operator) and (pred.att_idx == new_pred.att_idx):
                        if pred.operator == "<=":
                            pred.value = min(pred.value, new_pred.value)
                            rule.observed_class_distribution = best_suggestion. \
                                resulting_stats_from_split(0).copy()
                        elif pred.operator == ">":
                            pred.value = max(pred.value, new_pred.value)
                            rule.observed_class_distribution = best_suggestion. \
                                resulting_stats_from_split(1).copy()
                        rule._attribute_observers = {}
                        add_pred = False
                        break

                if add_pred:
                    rule.predicate_set.append(new_pred)
                    rule._attribute_observers = {}
                    rule.observed_class_distribution = {}
                    if new_pred.operator in ["=", "<="]:
                        rule.observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(0).copy()
                    else:
                        rule.observed_class_distribution = best_suggestion. \
                            resulting_stats_from_split(1).copy()

                    if self.expand_criterion == _FOILGAIN:
                        if not self.ordered_rules:
                            for c in rule.observed_class_distribution.keys():
                                if c != rule.class_idx:
                                    new_rule = copy.deepcopy(rule)
                                    new_rule.class_idx = c
                                    split_criterion = FoilGainExpandCriterion()
                                    should_expand = False
                                    best_expand_suggestions = new_rule.get_best_expand_suggestion(split_criterion, c)
                                    best_expand_suggestions.sort(key=attrgetter('merit'))
                                    if len(best_expand_suggestions) < 2:
                                        should_expand = len(best_expand_suggestions) > 0
                                    else:
                                        hoeffding_bound = self.compute_hoeffding_bound(
                                            split_criterion.get_range_of_merit(
                                                new_rule.observed_class_distribution), self.expand_confidence,
                                            new_rule.get_weight_seen())
                                        best_suggestion = best_expand_suggestions[-1]
                                        second_best_suggestion = best_expand_suggestions[-2]
                                        if ((best_suggestion.merit - second_best_suggestion.merit) > hoeffding_bound) \
                                                or (hoeffding_bound < self.tie_threshold):
                                            should_expand = True

                                        if self.remove_poor_atts is not None and self.remove_poor_atts:
                                            poor_atts = set()
                                            # Scan 1 - add any poor attribute to set
                                            for i in range(len(best_expand_suggestions)):
                                                if best_expand_suggestions[i] is not None:
                                                    split_atts = [best_expand_suggestions[i].att_idx]
                                                    if len(split_atts) == 1:
                                                        if best_suggestion.merit - best_expand_suggestions[i].merit > \
                                                                hoeffding_bound:
                                                            poor_atts.add(int(split_atts[0]))
                                            # Scan 2 - remove good attributes from set
                                            for i in range(len(best_expand_suggestions)):
                                                if best_expand_suggestions[i] is not None:
                                                    split_atts = [best_expand_suggestions[i].att_idx]
                                                    if len(split_atts) == 1:
                                                        if best_suggestion.merit - best_expand_suggestions[i].merit < \
                                                                hoeffding_bound:
                                                            try:
                                                                poor_atts.remove(int(split_atts[0]))
                                                            except KeyError:
                                                                pass
                                            for poor_att in poor_atts:
                                                new_rule.disable_attribute(poor_att)

                                    if should_expand:
                                        best_suggestion = best_expand_suggestions[-1]
                                        new_pred = Predicate(best_suggestion.att_idx, best_suggestion.operator,
                                                             best_suggestion.att_val)
                                        add_pred = True
                                        for pred in new_rule.predicate_set:
                                            if (pred.operator == new_pred.operator) and (
                                                    pred.att_idx == new_pred.att_idx):
                                                if pred.operator == "<=":
                                                    pred.value = min(pred.value, new_pred.value)
                                                    new_rule.observed_class_distribution = best_suggestion. \
                                                        resulting_stats_from_split(0).copy()
                                                elif pred.operator == ">":
                                                    pred.value = max(pred.value, new_pred.value)
                                                    new_rule.observed_class_distribution = best_suggestion. \
                                                        resulting_stats_from_split(1).copy()
                                                new_rule._attribute_observers = {}
                                                add_pred = False
                                                break
                                        if add_pred:
                                            new_rule.predicate_set.append(new_pred)
                                            new_rule._attribute_observers = {}
                                            new_rule.observed_class_distribution = {}
                                            if new_pred.operator in ["=", "<="]:
                                                new_rule.observed_class_distribution = best_suggestion. \
                                                    resulting_stats_from_split(0).copy()
                                            else:
                                                new_rule.observed_class_distribution = best_suggestion. \
                                                    resulting_stats_from_split(1).copy()
                                        self.rule_set.append(copy.deepcopy(new_rule))

    def predict(self, X):
        """Predicts the label of the instance(s).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)

    def predict_proba(self, X):
        """Predicts probabilities of all label of the instance(s).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = copy.deepcopy(self.get_votes_for_instance(X[i]))
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        return np.array(predictions)

    def get_model_measurements(self):
        """Collect metrics corresponding to the current status of the model.

        Returns
        -------
        string
            A string buffer containing the measurements of the model.
        """
        size = calculate_object_size(self)
        measurements = {'Number of rules: ': len(self.rule_set), 'model_size in bytes': size}
        return measurements

    def measure_model_size(self, unit='byte'):
        return calculate_object_size(self, unit)

    def reset(self):
        """ Resets the model to its initial state.

        Returns
        -------
        StreamModel
            self

        """
        self.rule_set = []
        self.default_rule = self.new_rule(None, None)
        self.classes = None
        return self

    def get_model_rules(self):
        """ Get the rules that describe the model

        Returns
        -------
        list (rule)
        """
        for rule in self.rule_set:
            class_idx = max(rule.observed_class_distribution.items(), key=itemgetter(1))[0]
            rule.class_idx = class_idx
        return self.rule_set

    def get_model_description(self):
        """ Returns the rules of the model

         Returns
         -------
         string
            Description of the rules
         """
        description = ''
        for i, rule in enumerate(self.rule_set):
            class_idx = max(rule.observed_class_distribution.items(), key=itemgetter(1))[0]
            description += 'Rule ' + str(i) + ' :' + str(rule.get_rule()) + '| class :' + str(class_idx) + '  ' + \
                           str(rule.observed_class_distribution) + '\n'
        class_idx = max(self.default_rule.observed_class_distribution.items(), key=itemgetter(1))[0]
        description += 'Default Rule :' + str(self.default_rule.get_rule()) + '| class :' + str(class_idx) + '  ' + \
                       str(self.default_rule.observed_class_distribution)
        return description

    @staticmethod
    def compute_hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each node.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an information gain the range
        is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct attribute at any given node.

        :math:`n`: Number of samples.

        Parameters
        ----------
        range_val: float
            Range value.
        confidence: float
            Confidence of choosing the correct attribute.
        n: float
            Number of samples.

        Returns
        -------
        float
            The Hoeffding bound.

        """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def __init__(self,
                 expand_confidence=0.0000001,
                 ordered_rules=True,
                 grace_period=200,
                 tie_threshold=0.05,
                 rule_prediction='first_hit',
                 nominal_attributes=None,
                 max_rules=1000,
                 nb_threshold=0,
                 nb_prediction=True,
                 drift_detector=None,
                 expand_criterion='info_gain',
                 remove_poor_atts=False,
                 min_weight=100):

        super().__init__()
        self.grace_period = grace_period
        self.expand_confidence = expand_confidence
        self.tie_threshold = tie_threshold
        self.rule_prediction = rule_prediction
        self.nominal_attributes = nominal_attributes
        self.max_rules = max_rules
        self.nb_threshold = nb_threshold
        self.ordered_rules = ordered_rules
        self.drift_detector = drift_detector
        self.expand_criterion = expand_criterion
        self.remove_poor_atts = remove_poor_atts
        self.nb_prediction = nb_prediction
        self.min_weight = min_weight

        self.rule_set = []
        self.default_rule = self.new_rule(None, None)
        self.classes = None

    @property
    def grace_period(self):
        return self._grace_period

    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period

    @property
    def expand_confidence(self):
        return self._expand_confidence

    @expand_confidence.setter
    def expand_confidence(self, expand_confidence):
        self._expand_confidence = expand_confidence

    @property
    def tie_threshold(self):
        return self._tie_threshold

    @tie_threshold.setter
    def tie_threshold(self, tie_threshold):
        self._tie_threshold = tie_threshold

    @property
    def remove_poor_atts(self):
        return self._remove_poor_atts

    @remove_poor_atts.setter
    def remove_poor_atts(self, remove_poor_atts):
        self._remove_poor_atts = remove_poor_atts

    @property
    def rule_prediction(self):
        return self._rule_prediction

    @rule_prediction.setter
    def rule_prediction(self, value):
        if value != _FIRSTHIT and value != _WEIGHTEDMAX \
                and value != _WEIGHTEDSUM:
            print("Invalid rule_prediction option '{}', will use '{}'".format(value, _FIRSTHIT))
            self._rule_prediction = _FIRSTHIT
        else:
            self._rule_prediction = value

    @property
    def nb_threshold(self):
        return self._nb_threshold

    @nb_threshold.setter
    def nb_threshold(self, nb_threshold):
        self._nb_threshold = nb_threshold

    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, value):
        if value is None:
            self._nominal_attributes = []
            print("No Nominal attributes have been defined, will consider all attributes as numerical")
        else:
            self._nominal_attributes = value

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    @property
    def ordered_rules(self):
        return self._ordered_rules

    @ordered_rules.setter
    def ordered_rules(self, value):
        if value and self.rule_prediction != _FIRSTHIT:
            print("Only one rule from the ordered set can be covered, rule prediction is set to first hit")
            self.rule_prediction = _FIRSTHIT
            self._ordered_rules = True
        else:
            self._ordered_rules = value
