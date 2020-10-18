import math

from river import proba


def entropy(dist: proba.Multinomial):
    """Returns the entropy of a multinomial distribution.

    Parameters:
        dist

    Example:

        >>> from river import proba

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial(events)

        >>> entropy(dist)
        1.97095

    References:
        1. [A Simple Explanation of Information Gain and Entropy](https://victorzhou.com/blog/information-gain/)
        2. [Calculating entropy <https://www.johndcook.com/blog/2013/08/17/calculating-entropy/)

    """
    return -sum(dist.pmf(c) * math.log2(dist.pmf(c)) for c in dist if dist.pmf(c) > 0)


def gini_impurity(dist: proba.Multinomial):
    """Returns the Gini impurity of a multinomial distribution.

    Parameters:
        dist

    Example:

        >>> from river import proba

        >>> events = [
        ...     'sunny', 'sunny',
        ...     'rainy', 'rainy', 'rainy',
        ...     'snowy', 'snomy', 'snowy', 'snomy', 'snowy'
        ... ]

        >>> dist = proba.Multinomial(events)

        >>> gini_impurity(dist)
        0.74

    References:
        1. [A Simple Explanation of Gini Impurity](https://victorzhou.com/blog/gini-impurity/)

    """
    return sum(dist.pmf(c) * (1 - dist.pmf(c)) for c in dist)
