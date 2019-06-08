import collections
import functools
import math
import operator


class Op(collections.namedtuple('Op', 'symbol operator')):

    def __call__(self, a, b):
        return self.operator(a, b)

    def __str__(self):
        return self.symbol


LT = Op('<', operator.lt)
EQ = Op('=', operator.eq)


class Split:
    """A data class for storing split details."""

    __slots__ = 'feature', 'operator', 'value'

    def __init__(self, feature, operator, value):
        self.feature = feature
        self.operator = operator
        self.value = value

    def __str__(self):
        return f'{self.feature} {self.operator} {self.value}'

    def test(self, x):
        return self.operator(x[self.feature], self.value)


def enum_unary(values):
    """

    Example:

        >>> for op, val in enum_unary(['a', 'b', 'c', 'd']):
        ...     print(op, val)
        = a
        = b
        = c
        = d

    """
    for val in values:
        yield EQ, val


def enum_contiguous(values):
    """

    Example:

        >>> for op, val in enum_contiguous([0, 1, 2, 3]):
        ...     print(op, val)
        < 1
        < 2
        < 3

    """
    for val in values[1:]:
        yield LT, val


def search_split_info_gain(class_counts, feature_counts, categoricals, criterion):

    best_gain = -math.inf
    second_best_gain = -math.inf
    split = None

    current_impurity = criterion(class_counts)

    for feature, counts in feature_counts.items():

        # Decide how to enumerate the splits
        split_enum = enum_unary if feature in categoricals else enum_contiguous

        for op, val in split_enum(sorted(counts.keys())):

            # 1. Build the counts according to the proposed split

            l_counts = collections.Counter()
            r_counts = collections.Counter()

            for v in counts:
                if op(v, val):
                    l_counts += counts[v]
                else:
                    r_counts += counts[v]

            # 2. Calculate the gain in impurity

            l_count, l_impurity = sum(l_counts.values()), criterion(l_counts)
            r_count, r_impurity = sum(r_counts.values()), criterion(r_counts)

            impurity = (l_count * l_impurity + r_count * r_impurity) / (l_count + r_count)
            gain = current_impurity - impurity

            # 3. Check if the split is better

            if gain > best_gain:
                best_gain, second_best_gain = gain, best_gain
                split = Split(feature, op, val)
            elif gain > second_best_gain:
                second_best_gain = gain

    return best_gain - second_best_gain, split


def sum_counters(counters):
    return functools.reduce(operator.add, counters, collections.Counter())
