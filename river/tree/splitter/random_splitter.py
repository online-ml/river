from __future__ import annotations

import abc
import random
import sys

from river import stats
from river.tree.splitter import Splitter
from river.tree.utils import BranchFactory


class RandomSplitter(Splitter):
    def __init__(self, seed, buffer_size):
        super().__init__()
        self.seed = seed
        self.buffer_size = buffer_size

        self.threshold = None
        self.stats = None

        self._rng = random.Random(self.seed)
        self._buffer = []

    def clone(self, new_params: dict | None = None, include_attributes=False):
        """Change the behavior of clone to allow copies to have a different rng."""

        new_params = new_params or {}
        new_params["seed"] = self._rng.randint(0, sys.maxsize)
        return super().clone(new_params, include_attributes)

    @abc.abstractmethod
    def _update_stats(self, branch, target_val, sample_weight):
        pass

    def cond_proba(self, att_val, class_val) -> float:
        """This attribute observer does not support probability density estimation."""
        raise NotImplementedError

    def update(self, att_val, target_val, sample_weight) -> Splitter:
        if self.threshold is None:
            if len(self._buffer) < self.buffer_size:
                self._buffer.append((att_val, target_val, sample_weight))
                return self

            mn = min(self._buffer, key=lambda t: t[0])[0]
            mx = max(self._buffer, key=lambda t: t[0])[0]

            self.threshold = self._rng.uniform(mn, mx)

            for a, t, w in self._buffer:
                self._update_stats(0 if a <= self.threshold else 1, t, w)
            self._buffer = None

            return self

        self._update_stats(0 if att_val <= self.threshold else 1, target_val, sample_weight)

        return self

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        post_split_dist = [self.stats[0], self.stats[1]]
        merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

        split_suggestion = BranchFactory(
            merit=merit,
            feature=att_idx,
            split_info=self.threshold,
            children_stats=post_split_dist,
        )

        return split_suggestion


class RegRandomSplitter(RandomSplitter):
    def __init__(self, seed, buffer_size):
        super().__init__(seed, buffer_size)
        self.stats = {0: stats.Var(), 1: stats.Var()}

    def _update_stats(self, branch, target_val, sample_weight):
        self.stats[branch].update(target_val, sample_weight)

    @property
    def is_target_class(self) -> bool:
        return False
