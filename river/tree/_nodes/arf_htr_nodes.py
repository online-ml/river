from .arf_htc_nodes import BaseRandomLearningNode
from .htr_nodes import LearningNodeMean
from .htr_nodes import LearningNodeModel
from .htr_nodes import LearningNodeAdaptive


class RandomLearningNodeMean(BaseRandomLearningNode, LearningNodeMean):
    """ARF learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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


class RandomLearningNodeModel(BaseRandomLearningNode, LearningNodeModel):
    """ARF learning Node for regression tasks that always use a learning model to provide
    responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed, leaf_model):
        super().__init__(
            stats,
            depth,
            attr_obs,
            attr_obs_params,
            max_features,
            seed,
            leaf_model=leaf_model,
        )


class RandomLearningNodeAdaptive(BaseRandomLearningNode, LearningNodeAdaptive):
    """ARF learning node for regression tasks that dynamically selects between the
    target mean and the output of a learning model to provide responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, max_features, seed, leaf_model):
        super().__init__(
            stats,
            depth,
            attr_obs,
            attr_obs_params,
            max_features,
            seed,
            leaf_model=leaf_model,
        )
