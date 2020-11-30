import bisect
import collections
import functools

from river import utils
from river.utils.histogram import Bin  # noqa

from .._attribute_test import NumericAttributeBinaryTest
from .._attribute_test import AttributeSplitSuggestion
from .attribute_observer import AttributeObserver


class NumericAttributeClassObserverHistogram(AttributeObserver):
    """Numeric attribute observer for classification tasks that discretizes features
    using histograms.


    Parameters
    ----------
    n_bins
        The maximum number of bins in the histogram.
    n_splits
        The number of split points to evaluate when querying for the best split
        candidate.
    """

    def __init__(self, n_bins: int = 256, n_splits: int = 32):
        super().__init__()
        self.n_bins = n_bins
        self.n_splits = n_splits
        self.hists = collections.defaultdict(functools.partial(utils.Histogram, max_bins=n_bins))

    def update(self, att_val, class_val, sample_weight):
        for _ in range(int(sample_weight)):
            self.hists[class_val].update(att_val)

        return self

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        if class_val not in self.hists:
            return 0.0

        total_weight = self.hists[class_val].n
        if not total_weight > 0:
            return 0.0

        i = bisect.bisect(self.hists[class_val], Bin(att_val, att_val, 1))

        if i < len(self.hists[class_val]):
            b = self.hists[class_val][i]
        else:  # att_val exceeds the range: take the last bin
            b = self.hists[class_val][-1]

        # Approximates the PDF of x by using the frequency in its corresponding
        # histogram bin
        if b.left == b.right:
            return b.count / total_weight
        else:
            return (b.count * (att_val - b.left) / (b.right - b.left)) / total_weight

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        best_suggestion = None

        low = min(h[0].right for h in self.hists.values())
        high = min(h[-1].right for h in self.hists.values())

        # If only one single value has been observed, then no split can be proposed
        if low >= high:
            return

        n_thresholds = min(self.n_splits, max(map(len, self.hists.values())) - 1)

        thresholds = list(decimal_range(start=low, stop=high, num=n_thresholds))
        cdfs = {y: hist.iter_cdf(thresholds) for y, hist in self.hists.items()}

        for at in thresholds:

            l_dist = {}
            r_dist = {}
            total_weight = sum(pre_split_dist.values())

            for y in pre_split_dist:
                p_xy = next(cdfs[y]) if y in cdfs else 0.0  # P(x < t | y)
                p_y = pre_split_dist[y] / total_weight  # P(y)
                l_dist[y] = total_weight * p_y * p_xy  # P(y | x < t)
                r_dist[y] = total_weight * p_y * (1 - p_xy)  # P(y | x >= t)

                post_split_dist = [l_dist, r_dist]
                merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

                if best_suggestion is None or merit > best_suggestion.merit:
                    num_att_binary_test = NumericAttributeBinaryTest(
                        att_idx, at, equal_passes_test=False
                    )
                    best_suggestion = AttributeSplitSuggestion(
                        num_att_binary_test, post_split_dist, merit
                    )

        return best_suggestion


def decimal_range(start, stop, num):
    """
    Example
    -------
    >>> for x in decimal_range(0, 1, 4):
    ...     print(x)
    0.2
    0.4
    0.6
    0.8
    """
    step = (stop - start) / (num + 1)

    for _ in range(num):
        start += step
        yield start
