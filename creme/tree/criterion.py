import math


def entropy(counts):
    """Computes entropy of a set of counts.

    If used by a decision tree learning algorithm, the goal is to minimize the entropy inside each
    leaf.

    Parameters:
        counts (collections.Counter)

    Example:

        >>> counts = {
        ...     'sunny': 4,
        ...     'rainy': 2,
        ...     'snowy': 10
        ... }
        >>> entropy(counts)
        1.298794...

    References:
        1. `A Simple Explanation of Information Gain and Entropy <https://victorzhou.com/blog/information-gain/>`_
        2. `Calculating entropy <https://www.johndcook.com/blog/2013/08/17/calculating-entropy/>`_

    """
    N = sum(counts.values())
    return math.log2(N) - sum(n * math.log2(n) for n in counts.values()) / N


def gini_impurity(counts):
    """Computes Gini impurity of a set of counts.

    If used by a decision tree learning algorithm, the goal is to minimize the Gini impurity inside
    each leaf.

    Parameters:
        counts (collections.Counter)

    Example:

        >>> counts = {
        ...     'sunny': 4,
        ...     'rainy': 2,
        ...     'green': 10
        ... }
        >>> gini_impurity(counts)
        0.53125...

    References:
        1. `A Simple Explanation of Gini Impurity <https://victorzhou.com/blog/gini-impurity/>`_

    """
    N = sum(counts.values())
    return sum(c / N * (1 - c / N) for c in counts.values())
