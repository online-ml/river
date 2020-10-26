import numbers
from river.utils.skmultiflow_utils import check_random_state

from .htc_nodes import LearningNodeMC
from .htc_nodes import LearningNodeNB
from .htc_nodes import LearningNodeNBA


class RandomLearningNodeMC(LearningNodeMC):
    """ARF learning node that always predicts the majority class.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """
    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []

    def update_attribute_observers(self, x, y, sample_weight, tree, **kwargs):
        def sample_features(instance, rng, max_features):
            selected = rng.choice(len(instance), size=max_features, replace=False)
            features = list(instance.keys())
            return [features[s] for s in selected]

        if len(self.feature_indices) == 0:
            self.feature_indices = sample_features(x, self._rng, self.max_features)

        for idx in self.feature_indices:
            if idx in self._disabled_attrs:
                continue

            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if ((tree.nominal_attributes is not None and idx in tree.nominal_attributes)
                        or not isinstance(x[idx], numbers.Number)):
                    obs = self.new_nominal_attribute_observer(**kwargs)
                else:
                    obs = self.new_numeric_attribute_observer(**kwargs)
                self.attribute_observers[idx] = obs
            obs.update(x[idx], y, sample_weight)


class RandomLearningNodeNB(LearningNodeNB):
    """ARF Naive Bayes learning node class.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []

    def update_attribute_observers(self, x, y, sample_weight, tree, **kwargs):
        def sample_features(instance, rng, max_features):
            selected = rng.choice(len(instance), size=max_features, replace=False)
            features = list(instance.keys())
            return [features[s] for s in selected]

        if len(self.feature_indices) == 0:
            self.feature_indices = sample_features(x, self._rng, self.max_features)

        for idx in self.feature_indices:
            if idx in self._disabled_attrs:
                continue

            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if ((tree.nominal_attributes is not None and idx in tree.nominal_attributes)
                        or not isinstance(x[idx], numbers.Number)):
                    obs = self.new_nominal_attribute_observer(**kwargs)
                else:
                    obs = self.new_numeric_attribute_observer(**kwargs)
                self.attribute_observers[idx] = obs
            obs.update(x[idx], y, sample_weight)


class RandomLearningNodeNBA(LearningNodeNBA):
    """ARF Naive Bayes Adaptive learning node class.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        Initial class observations.
    depth
        The depth of the node.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats, depth, max_features, seed):
        super().__init__(initial_stats, depth)
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []

    def update_attribute_observers(self, x, y, sample_weight, tree, **kwargs):
        def sample_features(instance, rng, max_features):
            selected = rng.choice(len(instance), size=max_features, replace=False)
            features = list(instance.keys())
            return [features[s] for s in selected]

        if len(self.feature_indices) == 0:
            self.feature_indices = sample_features(x, self._rng, self.max_features)

        for idx in self.feature_indices:
            if idx in self._disabled_attrs:
                continue

            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if ((tree.nominal_attributes is not None and idx in tree.nominal_attributes)
                        or not isinstance(x[idx], numbers.Number)):
                    obs = self.new_nominal_attribute_observer(**kwargs)
                else:
                    obs = self.new_numeric_attribute_observer(**kwargs)
                self.attribute_observers[idx] = obs
            obs.update(x[idx], y, sample_weight)