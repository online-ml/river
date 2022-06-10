import hashlib
import math

import numpy as np

from river import stats


class NUnique(stats.base.Univariate):
    """Approximate number of unique values counter.

    This is basically an implementation of the HyperLogLog algorithm. Adapted from
    [`hypy`](https://github.com/clarkduvall/hypy). The code is a bit too terse but it will do for
    now.

    Parameters
    ----------
    error_rate
        Desired error rate. Memory usage is inversely proportional to this value.
    seed
        Set the seed to produce identical results.

    Attributes
    ----------
    n_bits : int
    n_buckets : int
    buckets : list

    Examples
    --------

    >>> import string
    >>> from river import stats

    >>> alphabet = string.ascii_lowercase
    >>> n_unique = stats.NUnique(error_rate=0.2, seed=42)

    >>> n_unique.update('a').get()
    1

    >>> n_unique.update('b').get()
    2

    >>> for letter in alphabet:
    ...     n_unique = n_unique.update(letter)
    >>> n_unique.get()
    31

    Lowering the `error_rate` parameter will increase the precision.

    >>> n_unique = stats.NUnique(error_rate=0.01, seed=42)
    >>> for letter in alphabet:
    ...     n_unique = n_unique.update(letter)
    >>> n_unique.get()
    26

    References
    ----------
    [^1]: [My favorite algorithm (and data structure): HyperLogLog](https://odino.org/my-favorite-data-structure-hyperloglog/)
    [^2]: [Flajolet, P., Fusy, É., Gandouet, O. and Meunier, F., 2007, June. Hyperloglog: the analysis of a near-optimal cardinality estimation algorithm.](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)

    """

    P32 = 2**32

    def __init__(self, error_rate=0.01, seed: int = None):
        self.error_rate = error_rate
        self.seed = seed

        self.n_bits = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))
        self.n_buckets = 1 << self.n_bits
        self.buckets = [0] * self.n_buckets
        self._salt = np.random.RandomState(seed).bytes(hashlib.blake2s.SALT_SIZE)

    @property
    def name(self):
        return "n_unique"

    def _hash(self, x):
        hexa = hashlib.blake2s(bytes(x, encoding="utf8"), salt=self._salt).hexdigest()
        return int(hexa, 16)

    def update(self, x):
        x = self._hash(x)
        i = x & NUnique.P32 - 1 >> 32 - self.n_bits
        z = 35 - len(bin(NUnique.P32 - 1 & x << self.n_bits | 1 << self.n_bits - 1))
        self.buckets[i] = max(self.buckets[i], z)
        return self

    def get(self):
        a = (
            {16: 0.673, 32: 0.697, 64: 0.709}[self.n_buckets]
            if self.n_buckets <= 64
            else 0.7213 / (1 + 1.079 / self.n_buckets)
        )
        e = a * self.n_buckets * self.n_buckets / sum(1.0 / (1 << x) for x in self.buckets)
        if e <= self.n_buckets * 2.5:
            z = len([r for r in self.buckets if not r])
            return int(self.n_buckets * math.log(float(self.n_buckets) / z) if z else e)
        return int(e if e < NUnique.P32 / 30 else -NUnique.P32 * math.log(1 - e / NUnique.P32))
