import typing

from river.utils.skmultiflow_utils import check_random_state

from .htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from .leaf import HTLeaf


class BaseRandomLeaf(HTLeaf):
    """The Random Learning Node changes the way in which the attribute observers
    are updated (by using subsets of features).

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
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
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
        selected = self._rng.choice(len(x), size=max_features, replace=False)
        features = list(x.keys())
        return [features[s] for s in selected]


class RandomLeafMajorityClass(BaseRandomLeaf, LeafMajorityClass):
    """ARF learning node that always predicts the majority class.

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
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    kwargs
        Other parameters passed to the learning node.

    """

    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)


class RandomLeafNaiveBayes(BaseRandomLeaf, LeafNaiveBayes):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
        Number of attributes per subset for each node split.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)


class RandomLeafNaiveBayesAdaptive(BaseRandomLeaf, LeafNaiveBayesAdaptive):
    """ARF Naive Bayes Adaptive learning node class.

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
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, max_features, seed, **kwargs):
        super().__init__(stats, depth, splitter, max_features, seed, **kwargs)
