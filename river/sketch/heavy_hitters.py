from __future__ import annotations

import math
import operator
import typing

from river import base


class HeavyHitters(base.Base):
    """Find the Heavy Hitters using the Lossy Count with Forgetting factor algorithm[^1].

    Keep track of the most frequent item(set)s in a data stream and apply a forgetting factor to
    discard previous frequent items that do not often appear anymore. This is an approximation
    algorithm designed to work with a limited amount of memory rather than accounting for every possible
    solution (thus using an unbounded memory footprint). Any hashable type can be passed as input, hence
    tuples or frozensets can also be monitored.

    Considering a data stream where `n` elements were observed so far, the Lossy Count
    algorithm has the following properties:

    - All item(set)s whose true frequency exceeds `support * n` are output. There are no
    false negatives;

    - No item(set) whose true frequency is less than `(support - epsilon) * n` is outputted;

    - Estimated frequencies are less than the true frequencies by at most `epsilon * n`.

    Parameters
    ----------
    support
        The support threshold used to determine if an item is frequent. The value of `support` must
        be in $[0, 1]$. Elements whose frequency is higher than `support` times the number of
        observations seen so far are outputted.
    epsilon
        Error parameter to control the accuracy-memory tradeoff. The value of `epsilon` must be
        in $(0, 1]$ and typically `epsilon` $\\ll$ `support`. The smaller the `epsilon`, the more
        accurate the estimates will be, but the count sketch will have an increased memory
        footprint.
    fading_factor
        Forgetting factor applied to the frequency estimates to reduce the impact of old items.
        The value of `fading_factor` must be in $(0, 1]$.

    Examples
    --------

    >>> import random
    >>> import string
    >>> from river import sketch

    >>> rng = random.Random(42)
    >>> hh = sketch.HeavyHitters()

    We will feed the counter with printable ASCII characters:

    >>> for _ in range(10_000):
    ...     hh = hh.update(rng.choice(string.printable))

    We can retrieve estimates of the `n` top elements and their frequencies. Let's try `n=3`
    >>> hh.most_common(3)
    [(',', 122.099142...), ('[', 116.049510...), ('W', 115.013402...)]

    We can also access estimates of individual elements:

    >>> hh['A']
    99.483575...

    Unobserved elements are handled just fine:
    >>> hh[(1, 2, 3)]
    0.0

    References
    ----------
    [^1]: Veloso, B., Tabassum, S., Martins, C., Espanha, R., Azevedo, R., & Gama, J. (2020).
    Interconnect bypass fraud detection: a case study. Annals of Telecommunications, 75(9), 583-596.

    """

    def __init__(
        self, support: float = 0.001, epsilon: float = 0.005, fading_factor: float = 0.999
    ):
        if support > epsilon:
            raise ValueError("'support' must be smaller than 'epsilon'.")

        self.support = support
        self.epsilon = epsilon
        self.fading_factor = fading_factor

        self._bucket_width = math.ceil(1 / self.epsilon)
        self._n: int = 0
        self._entries: dict[typing.Hashable, tuple[float, float]] = {}
        self._delta: float = self._bucket_width

    def __getitem__(self, index) -> float:
        return self._entries.get(index, (0.0, None))[0]

    def update(self, x: typing.Hashable):
        self._n += 1
        current_bucket = math.ceil(self._n / self._bucket_width)
        freq, delta = 1.0, current_bucket - 1.0
        if x in self._entries:
            freq, delta = self._entries[x]
            freq += 1
        self._entries[x] = (freq, delta)

        # If at bucket boundary then prune low frequency entries.
        if self._n % self._bucket_width == 0:
            prune = []
            for key in self._entries:
                freq, delta = self._entries[key]
                freq *= self.fading_factor
                self._entries[key] = (freq, delta)

                if freq + delta <= current_bucket:
                    prune.append(key)

            for key in prune:
                del self._entries[key]

            self._delta = self._bucket_width + self._delta * self.fading_factor

        return self

    def most_common(self, n: int | None = None) -> list[tuple[typing.Hashable, float]]:
        res = []
        for key in self._entries:
            freq, _ = self._entries[key]
            if freq >= (self.support - self.epsilon) * self._delta:
                res.append((key, freq))

        if n is None:
            n = len(res)

        return sorted(res, key=operator.itemgetter(1), reverse=True)[:n]
