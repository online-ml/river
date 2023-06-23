from __future__ import annotations

import math
import random
import typing

from river import base


class Set(base.Base):
    """Approximate tracking of observed items using Bloom filters.

    Bloom filters enable using a limited amount of memory to check whether a given item was already
    observed in a stream. They can be used similarly to Python's built-in sets with the difference
    that items are not explicitly stored. For that reason, element removal and set difference are not
    currently supported.

    Bloom filters store a bit array and map incoming items to `k` index positions in the such array.
    The selected positions are set to `True`. Therefore, a binary code representation is created for each item.
    Membership works by projecting the query item and checking if every position of its binary code is
    `True`. If that is not the case, the item was not observed yet. A nice property of Bloom filters is
    that they do not yield false negatives: unobserved items might be signalized as observed, but observed
    items are never signalized as unobserved.

    If more than one item has the same binary code, i.e., hash collisions happen, the accuracy of the Bloom
    filter decreases, and false positives are produced. For instance, a previously unobserved item is signalized
    as observed. Increasing the size of the binary array and the value of `k` increase the filter's accuracy as
    hash collisions are avoided. Nonetheless, even using an increased number of hash functions, hash collisions
    will frequently happen if the array capacity is too small. The length of the bit array and the number of
    hash functions are inferred automatically from the supplied `capacity` and `fp_rate`.


    Parameters
    ----------
    capacity
        The maximum capacity of the Bloom filter, i.e., the maximum number of distinct items to store given
        the selected `fp_rate`.
    fp_rate
        The allowed rate of false positives. The probability of obtaining a true positive is `1 - fp_rate`.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> import random
    >>> from river import sketch

    >>> rng = random.Random(42)
    >>> s_set = sketch.Set(capacity=100, seed=0)


    We can retrieve the number of selected hash functions:

    >>> s_set.n_hash
    7

    And the size of the binary array used by the Bloom filter:
    >>> s_set.n_bits
    959

    We can add new items and check for membership using the same calls used by Python's
    standard sets:
    >>> for _ in range(1000):
    ...     s_set.add(rng.randint(0, 200))

    >>> 1 in s_set
    True

    False positives might happen if the capacity is not large enough:
    >>> -10 in s_set
    True

    Iterables can also be supplied to perform multiple updates with a single call to `update`:
    >>> s_set = s_set.update([1, 2, 3, 4, 5, 6, 7])

    We can also combine instances of `sketch.Set` using the intersection and union operations, as long as
    they share the same hash functions and capability. In other words, all they hyperparameters match.
    Let's create two instances that will monitor different portions of a stream of random numbers:

    >>> s1 = sketch.Set(seed=8)
    >>> s2 = sketch.Set(seed=8)

    >>> for _ in range(1000):
    ...     s1.add(rng.randint(0, 5000))

    >>> for _ in range(1000):
    ...     s2.add(rng.randint(0, 5000))

    >>> 43 in s1
    True
    >>> 43 in s2
    False

    We can get the intersection between the two instances by using:

    >>> s_intersection = s1 & s2
    >>> 43 in s_intersection
    False

    We can also obtain the set union:

    >>> s_union = s1 | s2

    >>> 43 in s_union
    True

    The same effect of the non-inplace dunder methods can be achieved via explicit method calls:

    >>> 43 in s1.intersection(s2)
    False

    >>> 43 in s1.union(s2)
    True

    Notes
    -----
    This implementation uses an integer to represent the binary array. Bitwise operations are performed in the
    integer to reflect the Bloom filter updates.

    References
    ----------
    [^1]: [Florian Hartmann's blog article on Bloom Filters](https://florian.github.io/bloom-filters/).
    [^2]: [Wikipedia entry on Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter).

    """

    def __init__(self, capacity: int = 2048, fp_rate: float = 0.01, seed: int | None = None):
        self.capacity = capacity
        self.fp_rate = fp_rate
        self.seed = seed

        # Size of the binary array
        self._asize = int(
            math.ceil(-((self.capacity * math.log(self.fp_rate)) / (math.log(2) ** 2)))
        )
        # Number of hash functions
        self._n_hash = int(math.ceil((self._asize / self.capacity) * math.log(2)))

        self._rng = random.Random(self.seed)

        # Random masks to obtain independent hash functions
        self._masks = [self._rng.getrandbits(64) for _ in range(self._n_hash)]

        # The binary array is represented as an integer that has with one extra bit.
        # The most significant bit is always 1 and it is used to ensure the bit_length
        # does not change during the execution.
        self._bloom = 2 ** (self._asize)

    @property
    def n_hash(self) -> int:
        """Return the number of used hash functions."""
        return self._n_hash

    @property
    def n_bits(self) -> int:
        """Return the size of the binary array used by the Bloom filter."""
        return self._asize

    def _hash(self, x: typing.Hashable):
        return [(hash(x) ^ self._masks[i]) % self._asize for i in range(self._n_hash)]

    def add(self, x: typing.Hashable):
        pos = self._hash(x)

        # Set the corresponding bits to 1
        # https://stackoverflow.com/a/12174125
        for p in pos:
            self._bloom |= 1 << p

    def update(self, values: typing.Iterable):
        for x in values:
            self.add(x)

        return self

    def __contains__(self, x: typing.Hashable):
        proj = []
        pos = self._hash(x)

        # Check if the corresponding bits are 1
        # https://stackoverflow.com/a/45221136
        for p in pos:
            proj.append((self._bloom >> p) & 1 == 1)

        return all(proj)

    def _is_mergeable(self, other: Set) -> bool:
        if len(self._masks) != len(other._masks):
            return False

        # If two instances share the same random masks, they also share the same
        # hash functions
        mask_check = all(m1 == m2 for m1, m2 in zip(self._masks, other._masks))
        return mask_check and self.capacity == other.capacity

    def _check_mergeable(self, other):
        if not self._is_mergeable(other):
            raise ValueError(
                "The supplied 'sketch.Set' instances cannot the combined.",
                "Ensure their 'capacity', 'fp_rate', and 'seed' match.",
            )

    def __iand__(self, other: Set):
        self._check_mergeable(other)

        self._bloom &= other._bloom
        return self

    def __and__(self, other: Set):
        new = self.clone(include_attributes=True)
        new &= other
        return new

    def __ior__(self, other: Set):
        self._check_mergeable(other)

        self._bloom |= other._bloom
        return self

    def __or__(self, other: Set):
        new = self.clone(include_attributes=True)
        new |= other

        return new

    def intersection(self, other: Set):
        """Set intersection.

        Return a new instance that results from the set intersection between the current `Set` object
        and `other`. Dunder operators can be used to replace the method call, i.e., `a &= b` and
        `a & b` for inplace and non-inplace intersections, respectively.

        Parameters
        ----------
        other
            Another instance of `sketch.Set`. All the hyperparameters values of `other` must match
            the values from the instance from which this method is called.

        Raises
        ------
        ValueError
            In case the hyperparameter values of the two involved `sketch.Set` instances do not match.
        """
        return self & other

    def union(self, other: Set):
        """Set union.

        Return a new instance that results from the set union between the current `Set` object and `other`.
        Dunder operators can be used to replace the method call, i.e., `a |= b` and `a | b` for inplace
        and non-inplace unions, respectively.

        Parameters
        ----------
        other
            Another instance of `sketch.Set`. All the hyperparameters values of `other` must match
            the values from the instance from which this method is called.

        Raises
        ------
        ValueError
            In case the hyperparameter values of the two involved `sketch.Set` instances do not match.
        """
        return self | other
