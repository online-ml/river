from __future__ import annotations

import random
import typing

from river.tree.nodes.htr_nodes import LeafAdaptive, LeafMean, LeafModel
from river.tree.nodes.leaf import HTLeaf


class ETLeaf(HTLeaf):
    """The Extra Tree leaves change the way in which the splitters are updated
    (by using subsets of features).

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    max_features
        Number of attributes per subset for each node split.
    seed
        Seed to ensure reproducibility.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.max_features = max_features
        self.seed = seed
        self._rng = random.Random(self.seed)
        self.feature_indices = []

    def _iter_features(self, x) -> typing.Iterable:
        # Only a random subset of the features is monitored
        if len(self.feature_indices) == 0:
            self.feature_indices = self._sample_features(x, self.max_features)

        for att_id in self.feature_indices:
            # First check if the feature is available
            if att_id in x:
                yield att_id, x[att_id]

    def _sample_features(self, x, max_features):
        return self._rng.sample(sorted(x.keys()), max_features)


class ETLeafMean(ETLeaf, LeafMean):
    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)


class ETLeafModel(ETLeaf, LeafModel):
    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)


class ETLeafAdaptive(ETLeaf, LeafAdaptive):
    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)
