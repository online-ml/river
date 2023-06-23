from __future__ import annotations

import abc
import copy
import math
import numbers
import typing

from river import base, tree


class Literal(base.Base):
    __slots__ = "on", "at", "neg"

    def __init__(self, on, at, neg=False):
        self.on = on
        self.at = at
        self.neg = neg

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def describe(self):
        pass


class NumericLiteral(Literal):
    def __init__(self, on, at, neg):
        super().__init__(on, at, neg)

    def __call__(self, x):
        if self.on in x:
            if not self.neg:
                return x[self.on] <= self.at
            else:
                return x[self.on] > self.at

        return False

    def describe(self):
        if not self.neg:
            return f"{self.on} ≤ {self.at:.4f}"
        else:
            return f"{self.on} > {self.at:.4f}"


class NominalLiteral(Literal):
    def __init__(self, on, at, neg):
        super().__init__(on, at, neg)

    def __call__(self, x):
        if self.on in x:
            if not self.neg:
                return x[self.on] == self.at
            else:
                return x[self.on] != self.at

        return False

    def describe(self):
        if not self.neg:
            return f"{self.on} = {self.at}"
        else:
            return f"{self.on} ≠ {self.at}"


class HoeffdingRule(base.Estimator, metaclass=abc.ABCMeta):
    """Base class for the decision rules based on the Hoeffding bound.

    It defines properties and operations shared by all the rule-based systems, for instance, how
    to perform rule expansion, how to monitor inputs, and whether or not a rule is currently
    covering an input datum.

    Parameters
    ----------
    template_splitter
        The attribute observer algorithm used to monitor and perform split attempts in numerical
        features. This splitter is be deep-copied to all the decision rules. The same attribute
        observer algorithms used by the Hoeffding Trees can be used with Hoeffding rules.
    split_criterion
        The criterion used to rank the split candidates. The same split criteria used by the
        Hoeffding Trees can be applied to the Hoeffding rules.
    attributes
        Other parameters passed to the rules via `**kwargs`.
    """

    def __init__(
        self,
        template_splitter: tree.splitter.base.Splitter,
        split_criterion: tree.split_criterion.base.SplitCriterion,
        **attributes,
    ):
        self.template_splitter = template_splitter
        self.split_criterion = split_criterion
        self.literals: list[Literal] = []
        self.splitters: dict[typing.Hashable, tree.splitter.Splitter] = {}

        self._total_weight = 0
        self._last_expansion_attempt_at = 0

        self.nominal_features: set[typing.Hashable] = set()

        self.__dict__.update(attributes)

    def _hoeffding_bound(self, r_heur, delta):
        r"""Compute the Hoeffding bound, used to decide how many samples are necessary to expand
        a decision rule.

        Notes
        -----
        The Hoeffding bound is defined as:

        $\\epsilon = \\sqrt{\\frac{R^2\\ln(1/\\delta))}{2n}}$

        where:

        $\\epsilon$: Hoeffding bound.
        $R$: Range of a random variable.
        $\\delta$: significance level.
        $n$: Number of samples.

        Parameters
        ----------
        r_heur
            Range of the split heuristic.
        delta
            The significance level.
        """
        return math.sqrt((r_heur * r_heur * math.log(1.0 / delta)) / (2.0 * self.total_weight))

    @property  # type: ignore
    @abc.abstractmethod
    def statistics(self):
        pass

    @statistics.setter  # type: ignore
    @abc.abstractmethod
    def statistics(self, target_stats):
        pass

    @property
    def total_weight(self):
        return self._total_weight

    @property
    def last_expansion_attempt_at(self):
        return self._last_expansion_attempt_at

    def expand(self, delta, tau):
        """
        Attempt to expand a decision rule.

        If the expansion succeeds, a new rule with all attributes reset and updated literals is
        returned along a boolean flag set to `True` to indicate the expansion. If the expansion
        fails, `self` is returned along with the boolean flag set to `False`.

        Parameters
        ----------
        delta
            The split test significance.
        tau
            The tie-breaking threshold.

        Returns
        -------
        rule, expanded
            The (potentially expanded) rule and an indicator of whether or not an expansion was
            successfully performed.

        """
        suggestions = []
        for att_id, splitter in self.splitters.items():
            suggestions.append(
                splitter.best_evaluated_split_suggestion(
                    criterion=self.split_criterion,
                    pre_split_dist=self.statistics,
                    att_idx=att_id,
                    binary_only=True,
                )
            )
        suggestions.sort()

        should_expand = False
        if len(suggestions) < 2:
            should_expand = True
        else:
            b_split = suggestions[-1]
            sb_split = suggestions[-2]

            hb = self._hoeffding_bound(self.split_criterion.range_of_merit(self.statistics), delta)

            if b_split.merit > 0 and (b_split.merit - sb_split.merit > hb or hb < tau):
                should_expand = True

        if should_expand:
            b_split = suggestions[-1]
            is_numerical = b_split.numerical_feature
            branch_no = self.split_criterion.select_best_branch(b_split.children_stats)

            if is_numerical:
                lit = NumericLiteral(b_split.feature, b_split.split_info, branch_no != 0)

                literal_updated = False
                for literal in self.literals:
                    if lit.on == literal.on and lit.neg == literal.neg:
                        # Update thresholds rather than adding a new literal
                        if not literal.neg and lit.at < literal.at:
                            literal.at = lit.at
                            literal_updated = True
                            break
                        elif literal.neg and lit.at > literal.at:
                            literal.at = lit.at
                            literal_updated = True
                            break

                # No threshold was updated, thus a new literal is added
                if not literal_updated:
                    self.literals.append(lit)
            else:
                lit = NominalLiteral(b_split.feature, b_split.split_info, branch_no != 0)
                # Add a new literal
                self.literals.append(lit)

            # Reset all the statistics stored in the decision rule
            updated_rule = self.clone()
            # Keep the literals
            updated_rule.literals.extend(self.literals)

            return updated_rule, True

        # If the expansion attempt failed, update the expansion tracker
        self._last_expansion_attempt_at = self.total_weight
        return self, False

    def covers(self, x: dict) -> bool:
        """Check if all the rule's conditions are fulfilled.

        Parameters
        ----------
        x
            Input instance.
        """
        return all(map(lambda lit: lit(x), self.literals))

    @abc.abstractmethod
    def new_nominal_splitter(self):
        """Define the attribute observer used when dealing with nominal features.

        Must be defined by classification and regression decision rule-based algorithms to match
        the target type.
        """
        pass

    def _iter_features(self, x: dict) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        yield from x.items()

    @abc.abstractmethod
    def _update_target_stats(self, y, w):
        pass

    def _update_feature_stats(self, feat_name, feat_val, w):
        # By default, no feature stats are kept
        pass

    def update(self, x, y, w):
        self._total_weight += w
        self._update_target_stats(y, w)

        for feat_name, feat_val in self._iter_features(x):
            try:
                splt = self.splitters[feat_name]
            except KeyError:
                if isinstance(feat_val, numbers.Number):
                    self.splitters[feat_name] = copy.deepcopy(self.template_splitter)
                else:
                    self.nominal_features.add(feat_name)
                    self.splitters[feat_name] = self.new_nominal_splitter()
                splt = self.splitters[feat_name]
            splt.update(feat_val, y, w)
            self._update_feature_stats(feat_name, feat_val, w)

    def __repr__(self):
        return f"{' and '.join([lit.describe() for lit in self.literals])}"
