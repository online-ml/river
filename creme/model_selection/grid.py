import itertools
import typing

import numpy as np


__all__ = ['expand_param_grid']


def expand_param_grid(grid: dict) -> typing.Iterator[dict]:
    """Expands a grid of parameters.

    This method can be used to generate a list of model parametrizations from a dictionary where
    each parameter is associated with a list of possible parameters. In other words, it expands a
    grid of parameters.

    Typically, this method can be used to create copies of a given model with different parameter
    choices. The models can then be used as part of a model selection process, such as a
    `model_selection.SuccessiveHalvingClassifier` or a `ensemble.HedgeRegressor`.

    The syntax for the parameter grid is quite flexible. It allows nesting parameters and can
    therefore be used to generate parameters for a pipeline.

    Parameters:
        grid: The grid of parameters to expand. The provided dictionary can be nested. The only
            requirement is that the values at the leaves need to be lists.

    Example:

        As an initial example, we can expand a grid of parameters for a single model.

        >>> import pprint
        >>> from creme import model_selection
        >>> from creme import optim

        >>> grid = {'optimizer': [optim.SGD(.1), optim.SGD(.01), optim.SGD(.001)]}

        >>> for params in model_selection.expand_param_grid(grid):
        ...     pprint.pprint(params)
        {'optimizer': SGD({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0})}
        {'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}
        {'optimizer': SGD({'lr': Constant({'learning_rate': 0.001}), 'n_iterations': 0})}

        You can parametrize a model with a given set of parameters with the `_set_params` method:

        >>> from creme import linear_model

        >>> model = linear_model.LinearRegression(optimizer=optim.SGD())
        >>> model = model._set_params(params)
        >>> model
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          intercept=0.
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

        You may specify a grid of parameters for a pipeline via nesting:

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

    """

    def expand_tuple(t):

        klass, params = t

        if not isinstance(klass, type):
            raise ValueError(f'Expected first element to be a class, got {klass}')

        if not isinstance(params, dict):
            raise ValueError(f'Expected second element to be a dict, got {params}')

        return (klass(**combo) for combo in expand_param_grid(params))

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
            return ((k, el) for el in expand_param_grid(v))

        raise ValueError(f'unsupported type: {type(v)}')

    for key in grid:
        if not isinstance(key, str):
            raise ValueError(f'Expected a key of type str; got {key}')

    # Example:
    # grid = {
    #     'beta_1': [0.1, 0.01, 0.001],
    #     'lr': [0.1, 0.01, 0.001]
    # }
    return (
        dict(el) if isinstance(el[0], tuple) else el[0]
        for el in itertools.product(*(expand(k, v) for k, v in grid.items()))
    )
