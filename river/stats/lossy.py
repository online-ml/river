import math
import operator
import typing

from river import stats


class LossyCount(stats.base.Univariate):
    """Lossy Count with Forgetting factor[^1].

    Keep track of the most frequent item(set)s in a data stream and apply a forgetting factor to
    discard previous frequent items that do not often appear anymore. This is an approximation
    algorithm designed to work with a limited amount of memory rather than accounting for every possible
    solution (thus using an unbounded memory footprint). Any hashable type can be passed as input, hence
    tuples or frozensets can also be monitored.

    Considering a data stream where `n` elements were observed so far, the Lossy Count
    algorithm has the following properties:

    - All item(set)s whose true frequency exceeds `support * n` are output. There are no
    false negatives

    - No item(set) whose true frequency is less than `(support - epsilon) * n` is outputted

    - Estimated frequencies are less than the true frequencies by at most `epsilon * n`

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
    alpha
        Forgetting factor applied to the frequency estimates to reduce the impact of old items.
        The value of `alpha` must be in $(0, 1]$.

    Examples
    --------

    >>> from river import datasets
    >>> from river import stats

    >>> dataset = datasets.TREC07()
    >>> lc = stats.LossyCount()

    >>> for x, _ in dataset.take(10_000):
    ...     lc = lc.update(x["sender"])

    >>> counts = lc.get()
    >>> for sender in counts:
    ...     print(f"{sender}\t{lc[sender]:.2f}")
    Groupe Desjardins / AccesD <services.de.cartes@scd.desjardins.com>    2494.79
    Groupe Desjardins / AccesD <securiteaccesd@desjardins.com>    77.82
    "AccuWeather.com Alert" <inbox@messaging.accuweather.com>    67.15
    <alert@broadcast.shareholder.com>    56.22
    "Bank of America Inc." <Security@bankofamerica.com>    28.37
    metze@samba.org    22.95
    tridge@samba.org    15.98
    Michael Adam <ma@sernet.de>    5.99
    abartlet@samba.org    4.00
    "Zachary Kline" <Z_kline@hotmail.com>    3.99
    Jonathan Worthington <jonathan@jnthn.net>    2.99
    charles loboz <charles_loboz@yahoo.com>    2.00
    slashdot@slashdot.org    2.00

    References
    ----------
    [^1]: Veloso, B., Tabassum, S., Martins, C., Espanha, R., Azevedo, R., & Gama, J. (2020).
    Interconnect bypass fraud detection: a case study. Annals of Telecommunications, 75(9), 583-596.

    """

    def __init__(self, support: float = 0.001, epsilon: float = 0.005, alpha: float = 0.999):
        if support > epsilon:
            raise ValueError("'support' must be smaller than 'epsilon'.")

        self.support = support
        self.epsilon = epsilon
        self.alpha = alpha

        self._bucket_width = math.ceil(1 / self.epsilon)
        self._n: int = 0
        self._entries: typing.Dict[typing.Hashable, typing.Tuple[float, float]] = {}
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
                freq *= self.alpha
                self._entries[key] = (freq, delta)

                if freq + delta <= current_bucket:
                    prune.append(key)

            for key in prune:
                del self._entries[key]

            self._delta = self._bucket_width + self._delta * self.alpha

        return self

    def get(self) -> typing.Optional[typing.List[typing.Hashable]]:  # type: ignore
        res = []
        for key in self._entries:
            freq, _ = self._entries[key]
            if freq >= (self.support - self.epsilon) * self._delta:
                res.append((key, freq))
        if res:
            return [elem[0] for elem in sorted(res, key=operator.itemgetter(1), reverse=True)]

    def __repr__(self):
        try:
            value = self.get()
        except NotImplementedError:
            value = None

        fmt_value = None
        if value is not None:
            fmt_value = " ".join([str(v) for v in value])

        return f"{self.__class__.__name__}: {fmt_value}"
