import itertools
import types
import typing

import numpy as np


__all__ = ['expand_param_grid']


def expand_param_grid(grid: dict) -> typing.Iterator[dict]:
    """Expands a grid of possible parameters into a sequence of single parametrizations.

    Parameters:
        grid: The grid of parameters to expand. The provided dictionary can be nested. The only
            requirement is that the values at the leaves need to be lists.

    Example:

        >>> import pprint
        >>> from creme import model_selection
        >>> from creme import optim

        >>> param_grid = {
        ...     'BagOfWords': {
        ...         'strip_accents': [False, True]
        ...     },
        ...     'LinearRegression': {
        ...         'intercept_lr': [0.001],
        ...         'optimizer': (optim.SGD, {'lr': [0.001, 0.01, 0.1]}),
        ...     }
        ... }

        >>> for params in model_selection.expand_param_grid(param_grid):
        ...     pprint.pprint(params)
        ...     print()
        {'BagOfWords': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.001}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'BagOfWords': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'BagOfWords': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'BagOfWords': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.001}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'BagOfWords': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'BagOfWords': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0})}}
        <BLANKLINE>

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
