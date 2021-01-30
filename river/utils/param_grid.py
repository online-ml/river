import itertools
import typing

import numpy as np

from river import base

__all__ = ["expand_param_grid"]


def expand_param_grid(model: base.Estimator, grid: dict) -> typing.List[base.Estimator]:
    """Expands a grid of parameters.

    This method can be used to generate a list of model parametrizations from a dictionary where
    each parameter is associated with a list of possible parameters. In other words, it expands a
    grid of parameters.

    Typically, this method can be used to create copies of a given model with different parameter
    choices. The models can then be used as part of a model selection process, such as a
    `expert.SuccessiveHalvingClassifier` or a `expert.EWARegressor`.

    The syntax for the parameter grid is quite flexible. It allows nesting parameters and can
    therefore be used to generate parameters for a pipeline.

    Parameters
    ----------
    model
    grid
        The grid of parameters to expand. The provided dictionary can be nested. The only
        requirement is that the values at the leaves need to be lists.

    Examples
    --------

    As an initial example, we can expand a grid of parameters for a single model.

    >>> from river import linear_model
    >>> from river import optim
    >>> from river import utils

    >>> model = linear_model.LinearRegression()

    >>> grid = {'optimizer': [optim.SGD(.1), optim.SGD(.01), optim.SGD(.001)]}
    >>> models = utils.expand_param_grid(model, grid)
    >>> len(models)
    3

    >>> models[0]
    LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.1
        )
      )
      loss=Squared ()
      l2=0.
      intercept_init=0.
      intercept_lr=Constant (
        learning_rate=0.01
      )
      clip_gradient=1e+12
      initializer=Zeros ()
    )

    You can expand parameters for multiple choices like so:

    >>> grid = {
    ...     'optimizer': [
    ...         (optim.SGD, {'lr': [.1, .01, .001]}),
    ...         (optim.Adam, {'lr': [.1, .01, .01]})
    ...     ]
    ... }
    >>> models = utils.expand_param_grid(model, grid)
    >>> len(models)
    6

    You may specify a grid of parameters for a pipeline via nesting:

    >>> from river import feature_extraction

    >>> model = (
    ...     feature_extraction.BagOfWords() |
    ...     linear_model.LinearRegression()
    ... )

    >>> grid = {
    ...     'BagOfWords': {
    ...         'strip_accents': [False, True]
    ...     },
    ...     'LinearRegression': {
    ...         'optimizer': [
    ...             (optim.SGD, {'lr': [.1, .01]}),
    ...             (optim.Adam, {'lr': [.1, .01]})
    ...         ]
    ...     }
    ... }

    >>> models = utils.expand_param_grid(model, grid)
    >>> len(models)
    8

    """

    return [model._set_params(params) for params in _expand_param_grid(grid)]


def _expand_param_grid(grid: dict) -> typing.Iterator[dict]:
    def expand_tuple(t):

        klass, params = t

        if not isinstance(klass, type):
            raise ValueError(f"Expected first element to be a class, got {klass}")

        if not isinstance(params, dict):
            raise ValueError(f"Expected second element to be a dict, got {params}")

        return (klass(**combo) for combo in _expand_param_grid(params))

    def expand(k, v):

        if isinstance(v, tuple):
            return ((k, el) for el in expand_tuple(v))

        # Example:
        # k = 'lr'
        # v = [0.001, 0.01, 0.1]
        if isinstance(v, (list, set, np.ndarray)):

            combos = []

            for el in v:
                if isinstance(el, tuple):
                    for combo in expand_tuple(el):
                        combos.append((k, combo))
                else:
                    combos.append((k, el))

            return combos

        if isinstance(v, dict):

            # Example:
            # k = 'LinearRegression'
            # v = {
            #     'intercept_lr': [0.001],
            #     'l2': [1],
            #     'optimizer': {
            #         optim.Adam: {
            #             'beta_1': [0.1, 0.01, 0.001],
            #             'lr': [0.1, 0.01, 0.001]
            #         },
            #      }
            # }
            return ((k, el) for el in _expand_param_grid(v))

        raise ValueError(f"unsupported type: {type(v)}")

    for key in grid:
        if not isinstance(key, str):
            raise ValueError(f"Expected a key of type str; got {key}")

    # Example:
    # grid = {
    #     'beta_1': [0.1, 0.01, 0.001],
    #     'lr': [0.1, 0.01, 0.001]
    # }
    return (
        dict(el) if isinstance(el[0], tuple) else el[0]
        for el in itertools.product(*(expand(k, v) for k, v in grid.items()))
    )
