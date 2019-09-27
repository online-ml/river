import math


def entropy(dist):
    """Returns the entropy of a multinomial distribution.

    Parameters:
        dist (proba.Multinomial)

    Example:

        >>> from creme import proba

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial(events)

        >>> entropy(dist)
        1.970950...

    References:
        1. `A Simple Explanation of Information Gain and Entropy <https://victorzhou.com/blog/information-gain/>`_
        2. `Calculating entropy <https://www.johndcook.com/blog/2013/08/17/calculating-entropy/>`_

    """
    # TODO: use the walrus operator
    return -sum(dist.pmf(c) * math.log2(dist.pmf(c)) for c in dist if dist.pmf(c) > 0)


def gini_impurity(dist):
    """Returns the Gini impurity of a multinomial distribution.

    Parameters:
        dist (proba.Multinomial)

    Example:

        >>> from creme import proba

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial(events)

        >>> gini_impurity(dist)
        0.74

    References:
        1. `A Simple Explanation of Gini Impurity <https://victorzhou.com/blog/gini-impurity/>`_

    """
    # TODO: use the walrus operator
    return sum(dist.pmf(c) * (1 - dist.pmf(c)) for c in dist)
