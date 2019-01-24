from . import function


__all__ = ['BiasAppender']


class BiasAppender(function.FunctionTransformer):
    """
    Example
    -------

        #!python
        >>> import creme

        >>> x = {'x': 42}
        >>> creme.preprocessing.BiasAppender().fit_one(x)
        {'x': 42, 'bias': 1.0}

    """

    def __init__(self, name='bias'):
        super().__init__(self._add_bias)
        self.name = name

    def _add_bias(self, x):
        x[self.name] = 1.
        return x
