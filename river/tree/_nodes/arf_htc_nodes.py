import numbers
from river.utils.skmultiflow_utils import check_random_state

from .htc_nodes import LearningNode
from .htc_nodes import LearningNodeMC
from .htc_nodes import LearningNodeNB
from .htc_nodes import LearningNodeNBA


class BaseRandomLearningNode(LearningNode):
    """The Random Learning Node changes the way in which the attribute observers
    are updated (by using subsets of features).

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    **kwargs
        Other parameters passed to the learning nodes the ARF implementations randomize.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed, **kwargs):
        super().__init__(stats, depth, attr_obs, attr_obs_params, **kwargs)  # noqa
        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)
        self.feature_indices = []

    def update_attribute_observers(self, x, y, sample_weight, nominal_attributes):
        if len(self.feature_indices) == 0:
            self.feature_indices = self._sample_features(x, self.max_features)

        for idx in self.feature_indices:
            if idx in self._disabled_attrs or idx not in x:
                continue

            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if (nominal_attributes is not None and idx in nominal_attributes) or not isinstance(
                    x[idx], numbers.Number
                ):
                    obs = self.new_nominal_attribute_observer()
                else:
                    obs = self.new_numeric_attribute_observer(
                        attr_obs=self.attr_obs, attr_obs_params=self.attr_obs_params
                    )
                self.attribute_observers[idx] = obs
            obs.update(x[idx], y, sample_weight)

    def _sample_features(self, x, max_features):
        selected = self._rng.choice(len(x), size=max_features, replace=False)
        features = list(x.keys())
        return [features[s] for s in selected]


class RandomLearningNodeMC(BaseRandomLearningNode, LearningNodeMC):
    """ARF learning node that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed):
        super().__init__(stats, depth, attr_obs, attr_obs_params, max_features, seed)


class RandomLearningNodeNB(BaseRandomLearningNode, LearningNodeNB):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed):
        super().__init__(stats, depth, attr_obs, attr_obs_params, max_features, seed)


class RandomLearningNodeNBA(BaseRandomLearningNode, LearningNodeNBA):
    """ARF Naive Bayes Adaptive learning node class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    max_features
        Number of attributes per subset for each node split.
    seed
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed):
        super().__init__(stats, depth, attr_obs, attr_obs_params, max_features, seed)
