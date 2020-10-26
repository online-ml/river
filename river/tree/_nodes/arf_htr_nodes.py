import numbers
from river.utils.skmultiflow_utils import check_random_state

from .htr_nodes import LearningNodeMean
from .htr_nodes import LearningNodeModel
from .htr_nodes import LearningNodeAdaptive


class RandomLearningNodeMean(LearningNodeMean):
    """ ARF learning Node for regression tasks that always use the average target
    value as response.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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
        self.feature_indices = []
        self.seed = seed
        self._rng = check_random_state(self.seed)

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


class RandomLearningNodeModel(LearningNodeModel):
    """ ARF learning Node for regression tasks that always use a learning model to provide
    responses.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, depth, leaf_model, max_features, seed):
        super().__init__(initial_stats, depth, leaf_model)
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


class RandomLearningNodeAdaptive(LearningNodeAdaptive):
    """ ARF learning node for regression tasks that dynamically selects between the
    target mean and the output of a learning model to provide responses.

    This node changes the way in which the attribute observers are updated
    (by using subsets of features).

    Parameters
    ----------
    initial_stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, depth, leaf_model, max_features, seed):
        super().__init__(initial_stats, depth, leaf_model)
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
