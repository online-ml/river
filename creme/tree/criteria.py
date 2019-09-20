import math


def entropy(dist):
    """Returns the entropy of a counter.

    If used by a decision tree learning algorithm, the goal is to minimize the entropy inside each
    leaf.

    Parameters:
        counts (dict)

    Example:

        >>> from creme import proba
        >>> from scipy import stats

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial()

        >>> for e in events:
        ...     dist = dist.update(e)

        >>> entropy(dist)
        1.970950...

        >>> stats.entropy([dist.pmf(c) for c in dist], base=2)
        1.970950...

    References:
        1. `A Simple Explanation of Information Gain and Entropy <https://victorzhou.com/blog/information-gain/>`_
        2. `Calculating entropy <https://www.johndcook.com/blog/2013/08/17/calculating-entropy/>`_

    """

    entro = 0.

    for c in dist:
        p = dist.pmf(c)
        if p:
            entro -= p * math.log2(p)

    return entro


def gini(dist):
    """Returns the Gini impurity of a counter.

    If used by a decision tree learning algorithm, the goal is to minimize the Gini impurity inside
    each leaf.

    Parameters:
        dist (proba.Multinomial)

    Example:

        >>> from creme import proba
        >>> from scipy import stats

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial()

        >>> for e in events:
        ...     dist = dist.update(e)

        >>> gini(dist)
        0.74

    References:
        1. `A Simple Explanation of Gini Impurity <https://victorzhou.com/blog/gini-impurity/>`_

    """
    return sum(dist.pmf(c) * (1 - dist.pmf(c)) for c in dist)
