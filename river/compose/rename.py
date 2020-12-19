from .. import base


__all__ = ["Renamer"]


class Renamer(base.Transformer):
    """Renames keys based on given parameters.

    Parameters
    ----------
    prefix
    suffix

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12}
    >>> compose.Renamer(prefix='prefix_', suffix='_suffix').transform_one(x)
    {'prefix_a_suffix': 42, 'prefix_b_suffix': 12}

    """

    def __init__(self, prefix=None, suffix=None):
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    def _rename(self, s):
        return self.prefix + s + self.suffix

    def transform_one(self, x):
        return {self._rename(i): xi for i, xi in x.items()}
