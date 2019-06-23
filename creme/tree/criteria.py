import math


def entropy(counts):
    """Returns the entropy of a counter.

    If used by a decision tree learning algorithm, the goal is to minimize the entropy inside each
    leaf.

    Parameters:
        counts (dict)

    Example:

        >>> from scipy import stats

        >>> counts = {
        ...     'sunny': 20,
        ...     'rainy': 30,
        ...     'snowy': 50,
        ...     'cloudy': 0
        ... }
        >>> entropy(counts)
        1.485475...

        >>> stats.entropy([c / 100 for c in counts.values()], base=2)
        1.485475...

    References:
        1. `A Simple Explanation of Information Gain and Entropy <https://victorzhou.com/blog/information-gain/>`_
        2. `Calculating entropy <https://www.johndcook.com/blog/2013/08/17/calculating-entropy/>`_

    """
    N = sum(counts.values())
    return math.log2(N) - sum(n * math.log2(n) if n else 0 for n in counts.values()) / N


def gini(counts):
    """Returns the Gini impurity of a counter.

    If used by a decision tree learning algorithm, the goal is to minimize the Gini impurity inside
    each leaf.

    Parameters:
        counts (dict)

    Example:

        >>> counts = {
        ...     'sunny': 20,
        ...     'rainy': 30,
        ...     'snowy': 50,
        ...     'cloudy': 0
        ... }
        >>> gini(counts)
        0.62

    References:
        1. `A Simple Explanation of Gini Impurity <https://victorzhou.com/blog/gini-impurity/>`_

    """
    N = sum(counts.values())
    return sum(n / N * (1 - n / N) for n in counts.values())
