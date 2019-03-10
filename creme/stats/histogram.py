import collections

from . import base


class Histogram(base.RunningStatistic):
    """Computes a running histogram.

    Attributes:
        maxbins (int) : Number of bins
        histogram (dict) : Running histogram
    Example:

    ::
        >>> import numpy as np
        >>> from creme import stats
        >>> np.random.seed(42)
        >>> mu, sigma = 0, 1
        >>> s = np.random.normal(mu, sigma, 10000)
        >>> hist = stats.Histogram(maxbins = 256)
        >>> for x in s:
        ...     _ = hist.update(x)
        >>> h = hist.get()
        >>> list_histo = []
        >>> for k,v in h.items():
        ...     list_histo += [float(k) for _ in range(v)]
        >>> np.median(list_histo)
        0.0038825005558860673
        >>> np.median(s)
        -0.0025949757928775408
        >>> np.percentile(list_histo,84)
        1.0008870801977727
        >>> np.percentile(s,84)
        0.9966166326412649


    References :
    1. `A Streaming Parallel Decision Tree Algorithm <http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`_
    2. `Streaming Approximate Histograms in Go <https://www.vividcortex.com/blog/2013/07/08/streaming-approximate-histograms/>`_
    3. `Go implementation <https://github.com/VividCortex/gohistogram>`_
    """
    def __init__(self, maxbins=256):
        self.maxbins = maxbins
        self.histogram = collections.Counter()

    @property
    def name(self):
        return 'histogram'

    def update(self, x):
        self.histogram.update([x])
        # Merge histogram
        while len(self.histogram) > self.maxbins:
            sorted_list_keys = sorted(list(self.histogram.keys()))
            # find nearest bin
            delta_key = [
                j - i
                for i, j in zip(sorted_list_keys[:-1], sorted_list_keys[1:])
            ]
            min_delta = min(delta_key)
            id_min_delta = delta_key.index(min_delta) + 1

            key_to_merge_right = sorted_list_keys[id_min_delta]
            key_to_merge_left = sorted_list_keys[id_min_delta - 1]

            # merge the two bins by summing their count and making a weighted average for each one

            total_count = self.histogram[key_to_merge_right] + self.histogram[
                key_to_merge_left]
            merged_key = key_to_merge_left * self.histogram[
                key_to_merge_left] + key_to_merge_right * self.histogram[key_to_merge_right]
            merged_key /= total_count
            [
                self.histogram.pop(key)
                for key in [key_to_merge_right, key_to_merge_left]
            ]
            self.histogram.update({merged_key: total_count})

        return self

    def get(self):
        return dict(sorted(self.histogram.items()))

