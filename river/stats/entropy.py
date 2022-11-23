import collections
import math

from river import stats


class Entropy(stats.base.Univariate):
    """Running entropy.

    Parameters
    ----------
    fading_factor
        Fading factor.
    eps
        Small value that will be added to the denominator to avoid division by zero.

    Attributes
    ----------
    entropy : float
        The running entropy.
    n : int
        The current number of observations.
    counter : collections.Counter
        Count the number of times the values have occurred

    Examples
    --------

    >>> import math
    >>> import random
    >>> import numpy as np
    >>> from scipy.stats import entropy
    >>> from river import stats

    >>> def entropy_list(labels, base=None):
    ...   value,counts = np.unique(labels, return_counts=True)
    ...   return entropy(counts, base=base)

    >>> SEED = 42 * 1337
    >>> random.seed(SEED)

    >>> entro = stats.Entropy(fading_factor=1)

    >>> list_animal = []
    >>> for animal, num_val in zip(['cat', 'dog', 'bird'],[301, 401, 601]):
    ...     list_animal += [animal for i in range(num_val)]
    >>> random.shuffle(list_animal)

    >>> for animal in list_animal:
    ...     _ = entro.update(animal)

    >>> print(f'{entro.get():.6f}')
    1.058093
    >>> print(f'{entropy_list(list_animal):.6f}')
    1.058093

    References
    ----------
    [^1]: [Sovdat, B., 2014. Updating Formulas and Algorithms for Computing Entropy and Gini Index from Time-Changing Data Streams. arXiv preprint arXiv:1403.6348.](https://arxiv.org/pdf/1403.6348.pdf)

    """

    def __init__(self, fading_factor=1, eps=1e-8):

        if 0 < fading_factor <= 1:
            self.fading_factor = fading_factor
        else:
            raise ValueError("fading_factor must be between 0 excluded and 1")
        self.eps = eps
        self.entropy = 0
        self.n = 0
        self.counter = collections.Counter()

    @property
    def name(self):
        return "entropy"

    def update(self, x):

        cx = self.counter.get(x, 0)
        n = self.n
        eps = self.eps
        fading_factor = self.fading_factor

        entropy = self.entropy
        entropy = (n + eps) / (n + 1) * (fading_factor * entropy - math.log((n + eps) / (n + 1)))
        entropy -= (cx + 1) / (n + 1) * math.log((cx + 1) / (n + 1))
        entropy += (cx + eps) / (n + 1) * math.log((cx + eps) / (n + 1))
        self.entropy = entropy

        self.n += 1
        self.counter.update([x])

        return self

    def get(self):
        return self.entropy
