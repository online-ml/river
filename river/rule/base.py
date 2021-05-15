import abc
import copy
import math
import numbers
import typing

from river import base


class Literal(base.Base):
    __slots__ = "on, at, neg"

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
            return f"{self.on} ≤ {self.at}"
        else:
            return f"{self.on} > {self.at}"


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
    def __init__(self, template_splitter, split_criterion, **attributes):
        self.template_splitter = template_splitter
        self.split_criterion = split_criterion
        self.literals = []
        self.splitters = {}

        self._last_expansion_attempt_at = 0

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
        return math.sqrt(
            (r_heur * r_heur * math.log(1.0 / delta)) / (2.0 * self.total_weight)
        )

    @property
    @abc.abstractmethod
    def statistics(self):
        pass

    @statistics.setter
    @abc.abstractmethod
    def statistics(self, target_stats):
        pass

    @property
    @abc.abstractmethod
    def total_weight(self):
        pass

    @property
    def last_expansion_attempt_at(self):
        return self._last_expansion_attempt_at

    def expand(self, delta, tau):
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

            hb = self._hoeffding_bound(
                self.split_criterion.range_of_merit(self.statistics), delta
            )

            # TODO (1) implement Variance Ratio split criterion to ensure the following expression
            # TODO (2) holds for both classification and regression
            if b_split.merit - sb_split.merit > hb or hb < tau:
                should_expand = True

        if should_expand:
            b_split = suggestions[-1]

            is_numerical = b_split.is_numerical

            # TODO select which of the binary branches of the test will be kept
            branch_no = self.split_criterion.select_best_branch(b_split.children_stats)

            if is_numerical:
                lit = NumericLiteral(
                    b_split.feature, b_split.split_info, branch_no != 0
                )
            else:
                lit = NominalLiteral(
                    b_split.feature, b_split.split_info, branch_no != 0
                )

            # Add a new literal
            self.literals.append(lit)

            # Set the rule's statistics to match the data partition that was just performed
            self.statistics = b_split.children_stats[branch_no]

        self._last_expansion_attempt_at = self.total_weight
        return should_expand

    def covers(self, x):
        return all(map(lambda lit: lit(x), self.literals))

    @abc.abstractmethod
    def new_nominal_splitter(self):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        for att_id, att_val in x.items():
            yield att_id, att_val

    @abc.abstractmethod
    def _update_target_stats(self, y, w):
        pass

    def _update_feature_stats(self, feat_name, feat_val, w):
        # By default, no feature stats are kept
        pass

    def update(self, x, y, w):
        self._update_target_stats(y, w)

        for feat_name, feat_val in self._iter_features(x):
            self._update_feature_stats(feat_name, feat_val, w)
            try:
                splt = self.splitters[feat_name]
            except KeyError:
                if isinstance(feat_val, numbers.Number):
                    self.splitters[feat_name] = copy.deepcopy(self.template_splitter)
                else:
                    self.splitters[feat_name] = self.new_nominal_splitter()
                splt = self.splitters[feat_name]
            splt.update(x, y, w)

    def __repr__(self):
        return f"{' and '.join([lit.describe() for lit in self.literals])}"
