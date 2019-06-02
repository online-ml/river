import collections
import functools
import math
import operator

from .. import utils


class Split:

    __slots__ = 'feature', 'operator', 'value'

    def __init__(self, feature, operator, value):
        self.feature = feature
        self.operator = operator
        self.value = value

    def __str__(self):
        return f'{self.feature} {self.operator.__name__} {self.value}'

    def test(self, x):
        return self.operator(x[self.feature], self.value)


def enum_unary(values):
    """

    Example:

        >>> for op, val in enum_unary(['a', 'b', 'c', 'd']):
        ...     print(op.__name__, val)
        eq a
        eq b
        eq c
        eq d

    """
    for val in values:
        yield operator.eq, val


def enum_contiguous(values):
    """

    Example:

        >>> for op, val in enum_contiguous([0, 1, 2, 3]):
        ...     print(op.__name__, val)
        lt 1
        lt 2
        lt 3

    """
    for val in values[1:]:
        yield operator.lt, val


def search_split_info_gain(class_counts, feature_counts, categoricals):

    best_gain = -math.inf
    second_best_gain = -math.inf
    split = None

    current_entropy = utils.entropy(class_counts)

    for feature, counts in feature_counts.items():

        # Decide how to enumerate the splits
        split_enum = enum_unary if feature in categoricals else enum_contiguous

        for op, val in split_enum(sorted(counts.keys())):

            left_counts = collections.Counter()
            right_counts = collections.Counter()

            for v in counts:
                if op(v, val):
                    left_counts += counts[v]
                else:
                    right_counts += counts[v]

            left_total = sum(left_counts.values())
            right_total = sum(right_counts.values())

            entropy = left_total * utils.entropy(left_counts) + \
                right_total * utils.entropy(right_counts)
            entropy /= (left_total + right_total)

            gain = current_entropy - entropy

            if gain > best_gain:
                best_gain, second_best_gain = gain, best_gain
                split = Split(feature, op, val)
            elif gain > second_best_gain:
                second_best_gain = gain

    return best_gain - second_best_gain, split


def sum_counters(counters):
    return functools.reduce(operator.add, counters, collections.Counter())
