import itertools
import types


__all__ = ['expand_param_grid']


def expand_param_grid(grid):
    """Expands a grid of possible parameters into a list of single parametrizations.

    Parameter:
        grid (dict)

    Example:

        >>> import pprint
        >>> from creme import optim

        >>> param_grid = {
        ...     'CountVectorizer': {
        ...         'strip_accents': (False, True)
        ...     },
        ...     'LinearRegression': {
        ...         'intercept_lr': [0.001],
        ...         'optimizer': {
        ...             optim.SGD: {
        ...                 'lr': [0.001, 0.01, 0.1],
        ...             },
        ...         }
        ...     }
        ... }

        >>> for params in expand_param_grid(param_grid):
        ...     pprint.pprint(params)
        ...     print()
        {'CountVectorizer': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.001}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'CountVectorizer': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'CountVectorizer': {'strip_accents': False},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'CountVectorizer': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.001}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'CountVectorizer': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}}
        <BLANKLINE>
        {'CountVectorizer': {'strip_accents': True},
         'LinearRegression': {'intercept_lr': 0.001,
                              'optimizer': SGD({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0})}}

    """

    def expand(k, v):

        # Example:
        # k = 'lr'
        # v = [0.001, 0.01, 0.1]
        if isinstance(v, (list, tuple, set)):
            return ((k, el) for el in v)

        if isinstance(v, dict):

            # Example:
            # k = optim.Adam
            # v = {
            #     'beta_1': [0.8],
            #     'lr': [0.001, 0.01, 0.1],
            #     'beta_2': [0.9, 0.99, 0.42]
            # }
            if isinstance(k, types.ClassType):
                return (k(**el) for el in expand_param_grid(v))

            # Example:
            # k = 'LinearRegression'
            # v = {
            #     'intercept_lr': [0.001],
            #     'l2': [1],
            #     'optimizer': {
            #         optim.Adam: {
            #             'beta_1': [0.8],
            #             'lr': [0.001, 0.01, 0.1],
            #             'beta_2': [0.9, 0.99, 0.42]
            #         },
            #      }
            # }
            return ((k, el) for el in expand_param_grid(v))

        raise ValueError(f'unsupported type: {type(v)}')

    return (
        dict(el) if isinstance(el[0], tuple) else el[0]
        for el in itertools.product(*(expand(k, v) for k, v in grid.items()))
    )
