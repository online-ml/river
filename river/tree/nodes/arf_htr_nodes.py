from .arf_htc_nodes import BaseRandomLeaf
from .htr_nodes import LeafAdaptive, LeafMean, LeafModel


class RandomLeafMean(BaseRandomLeaf, LeafMean):
    """ARF learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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


class RandomLeafModel(BaseRandomLeaf, LeafModel):
    """ARF learning Node for regression tasks that always use a learning model to provide
    responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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
    leaf_model
        A `base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(
        self, stats, depth, splitter, max_features, seed, leaf_model, **kwargs
    ):
        super().__init__(
            stats, depth, splitter, max_features, seed, leaf_model=leaf_model, **kwargs
        )


class RandomLeafAdaptive(BaseRandomLeaf, LeafAdaptive):
    """ARF learning node for regression tasks that dynamically selects between the
    target mean and the output of a learning model to provide responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
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
    leaf_model
        A `base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(
        self, stats, depth, splitter, max_features, seed, leaf_model, **kwargs
    ):
        super().__init__(
            stats, depth, splitter, max_features, seed, leaf_model=leaf_model, **kwargs
        )
