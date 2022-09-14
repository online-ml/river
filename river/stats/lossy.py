import math
import typing
from operator import itemgetter

from river import stats


class LossyCount(stats.base.Univariate):
    """Lossy Counting with Forgetting factor[^1].

    Keep track of the frequent itemsets in a data stream and apply a forgetting factor to
    discard previous frequent items that do not often appear anymore.

    Considering a data stream where `N` elements were observed so far, the Lossy Counting
    algorithm has the following properties:

    - All item(set)s whose true frequency exceeds `support * N` are output. There are no
    false negatives;

    - No item(set) whose true frequency is less than `(support - epsilon) * N` is outputted;


    - Estimated frequencies are less than the true frequencies by at most `epsilon * N`.

    Parameters
    ----------
    support
        The support threshold used to determine if an item is frequent. The value of `support` must
        be in $[0,1]$.
    epsilon
        Error parameter to control the accuracy-memory tradeoff. The value of `epsilon` must be
        in $(0,1]$ and typically `epsilon` $\\ll$ `support`. The smaller the `epsilon`, the more
        accurate the estimates will be, but the count sketch will have an increased memory
        footprint.
    forgetting
        Forgetting factor applied to the frequency estimates to reduce the impact of old items.
        The value of `forgetting` must be in $(0,1]$.

    Examples
    --------

    >>> from river import datasets
    >>> from river import stats

    >>> dataset = datasets.TREC07()
    >>> lc = stats.LossyCount()

    >>> for x, _ in dataset.take(10_000):
    ...     lc.update(x["sender"])

    >>> counts = lc.get()
    >>> for sender, freq in counts:
    ...     print(f"{sender}\t{freq:.2f}")
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

    def __init__(self, support: float = 0.001, epsilon: float = 0.005, forgetting: float = 0.999):
        if support > epsilon:
            raise ValueError("'support' must be smaller than 'epsilon'.")

        self.support = support
        self.epsilon = epsilon
        self.forgetting = forgetting

        self._bucket_width = math.ceil(1 / self.epsilon)
        self._n: int = 0
        self._entries: typing.Dict[typing.Hashable, typing.Tuple[float, float]] = {}
        self._delta: float = self._bucket_width

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
                freq *= self.forgetting
                self._entries[key] = (freq, delta)

                if freq + delta <= current_bucket:
                    prune.append(key)

            for key in prune:
                del self._entries[key]

            self._delta = self._bucket_width + self._delta * self.forgetting

    def get(self):
        res = []
        for key in self._entries:
            freq, _ = self._entries[key]
            if freq >= (self.support - self.epsilon) * self._delta:
                res.append((key, freq))
        return sorted(res, key=itemgetter(1), reverse=True)
