import collections

from . import base


class Histogram(base.RunningStatistic):

    def __init__(self, maxbins=80):
        self.maxbins = maxbins
        self.total = 0
        self.histogram = collections.Counter()

    @property
    def name(self):
        return 'histogram'

    def update(self, x):
        self.histogram.update([x])
        self.histogram = self.merge(self.histogram, self.maxbins)
        return self

    @classmethod
    def merge(cls, histogram, maxbins):
        sorted_list_keys = sorted(list(histogram.keys()))

        while len(histogram) > maxbins:
            delta_key = [
                j - i
                for i, j in zip(sorted_list_keys[:-1], sorted_list_keys[1:])
            ]
            min_delta = min(delta_key)
            id_min_delta = delta_key.index(min_delta) + 1

            key_to_merge = sorted_list_keys[id_min_delta]
            key_to_merge_bis = sorted_list_keys[id_min_delta - 1]

            total_count = histogram[key_to_merge] + histogram[key_to_merge_bis]
            merged_key = key_to_merge_bis * histogram[
                key_to_merge_bis] + key_to_merge * histogram[key_to_merge]
            merged_key /= total_count
            [histogram.pop(key) for key in [key_to_merge, key_to_merge_bis]]
            histogram.update({merged_key: total_count})
            
        return histogram

    def get(self):
        return self.histogram
