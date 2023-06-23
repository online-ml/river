from __future__ import annotations

import math
import random
import typing

import numpy as np

from river import base


class Counter(base.Base):
    """Counting using the Count-Min Sketch (CMS) algorithm.

    Contrary to an exhaustive approach, e.g., using a `collections.Counter`, CMS uses a
    limited and fixed amount of memory. The CMS algorithm uses a sketch structure consisting of a
    matrix $w \\times d$.

    These dimensions are obtained via:

    - $w = \\lceil \\frac{e}{\\epsilon} \\rceil$, where $e$ is the Euler number.

    - $d = \\lceil \\ln\\left(\\frac{1}{\\delta} \\right) \\rceil$.

    Decreasing the values of $\\epsilon$ (`epsilon`) and $\\delta$ (`delta`) increase the accuracy of the
    algorithm, at the cost of increased memory usage. The values of `w` and `d` control the hash tables' capability
    and the amount of hash collisions, respectively.

    CMS works by keeping `d` hash tables with `w` slots each. Elements are mapped to a slot in each hash table.
    These tables store the counting estimates. This implementation assumes the turnstile case described in the paper,
    i.e., count values and updates can be negative.

    The count values obtained by CMS are always overestimates. Suppose $c_i$ and $\\hat{c}_i$ are the ground truth and
    estimated count values, respectively, for a given element $i$. CMS guarantees that $c_i \\le \\hat{c}_i$ and,
    with probability $1 - \\delta$, $\\hat{c}_i \\le c_i + \\epsilon||\\mathbf{c}||_1$. In the expression,
    $||\\mathbf{c}||_1 = \\sum_i |c_i|$.

    Parameters
    ----------
    epsilon
        The approximation error parameter. The error in answering a query is within a factor of `epsilon`
        with probability `delta`.
    delta
        A query estimates have a probability of `1 - delta` of having errors which are a factor of `epsilon`.
        See the CMS description above for more details.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> import collections
    >>> from river import sketch

    >>> cms = sketch.Counter(epsilon=0.005, seed=0)

    >>> # To generate random numbers
    >>> rng = random.Random(7)

    >>> counter = collections.Counter()

    We can check the number of slots per hash table:
    >>> cms.n_slots
    544

    And the number of hash tables:
    >>> cms.n_tables
    3

    Let's compare the sketch against a brute force approach:

    >>> vals = []
    >>> for _ in range(10000):
    ...     v = rng.randint(-1000, 1000)
    ...     cms = cms.update(v)
    ...     counter[v] += 1
    ...     vals.append(v)

    Now, we can compare the estimates of CMS against the exhaustive counting strategy:

    >>> counter[7]
    5
    >>> cms[7]
    12
    >>> counter[532]
    4
    >>> cms[532]
    15

    Keep in mind that CMS is an approximate sketch algorithm. Couting estimates for unseen values
    might not be always reliable:

    >>> cms[1001]
    9

    We can check the number of elements stored by each approach:

    >>> len(counter), len(cms)
    (1982, 1632)

    And also retrieve the total sum of counts:

    >>> cms.total()
    10000

    We can decrease the error by allocating more memory in the CMS:

    >>> cms_a = sketch.Counter(epsilon=0.001, delta=0.01, seed=0)
    >>> for v in vals:
    ...     cms_a = cms_a.update(v)

    >>> cms_a[7]
    5
    >>> cms_a[532]
    4

    We can also obtain estimates of the dot product between two instances of `river.collections.Counter`. This could be useful,
    for instance, to estimate the cosine distance between the data monitored in two different counter sketch instances. Suppose we
    create another CMS instance (the number of slots and hash tables must match) that monitors another sample of the same data
    generating process:

    >>> cms_b = sketch.Counter(epsilon=0.001, delta=0.01, seed=7)

    >>> for _ in range(10000):
    ...     v = rng.randint(-1000, 1000)
    ...     cms_b = cms_b.update(v)

    Now, we can define a cosine distance function:

    >>> def cosine_dist(cms_a, cms_b):
    ...     num = cms_a @ cms_b
    ...     den = math.sqrt(cms_a @ cms_a) * math.sqrt(cms_b @ cms_b)
    ...     return num / den

    And use it to calculate the cosine distance between the elements monitored in `cms_a` and `cms_b`:

    >>> cosine_dist(cms_a, cms_b)
    0.175363...

    References
    ----------
    [^1]: [Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.](https://www.cse.unsw.edu.au/~cs9314/07s1/lectures/Lin_CS9314_References/cm-latin.pdf)
    [^2]: [Count-Min Sketch](https://florian.github.io//count-min-sketch/)
    [^3]: [Hash functions family generator in Python](https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python)

    """

    def __init__(self, epsilon: float = 0.1, delta: float = 0.05, seed: int | None = None):
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed

        self._w = int(math.ceil(math.e / self.epsilon))
        self._d = int(math.ceil(math.log(1 / self.delta)))

        self._rng = random.Random(self.seed)
        self._masks = [self._rng.getrandbits(64) for _ in range(self._d)]
        self._cms = np.zeros((self._d, self._w), dtype=np.int32)

    def _hash(self, x):
        return tuple(zip(*((i, (hash(x) ^ self._masks[i]) % self._w) for i in range(self._d))))

    def __getitem__(self, x) -> int:
        # Point query
        return min(self._cms[self._hash(x)])

    def __matmul__(self, other: Counter) -> int:
        # Dot product
        if self.n_slots != other.n_slots or self.n_tables != other.n_tables:
            unmatched_dims = (
                f"({self.n_slots}, {self.n_tables}) vs. ({other.n_slots}, {other.n_tables})"
            )
            raise ValueError(f"The number of slots and hash tables do not match: {unmatched_dims}.")

        return min(np.einsum("ij,ij->i", self._cms, other._cms))

    def __len__(self):
        # Return the total number of stored elements in the sketch
        return self._w * self._d

    def update(self, x: typing.Hashable, w: int = 1):
        self._cms[self._hash(x)] += w

        return self

    def total(self) -> int:
        """Return the total count."""
        return sum(self._cms[0, :])

    @property
    def n_slots(self) -> int:
        """The number of slots in each hash table."""
        return self._w

    @property
    def n_tables(self) -> int:
        """The number of stored hash tables."""
        return self._d
