from __future__ import annotations

from river import base

__all__ = ["Renamer", "Prefixer", "Suffixer"]


class Renamer(base.Transformer):
    """Renames features following substitution rules.

    Parameters
    ----------
    mapping
        Dictionnary describing substitution rules. Keys in `mapping` that are not a feature's name are silently ignored.

    Examples
    --------

    >>> from river import compose

    >>> mapping = {'a': 'v', 'c': 'o'}
    >>> x = {'a': 42, 'b': 12}
    >>> compose.Renamer(mapping).transform_one(x)
    {'b': 12, 'v': 42}

    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def transform_one(self, x):
        for old_key, new_key in self.mapping.items():
            try:
                x[new_key] = x.pop(old_key)
            except KeyError:
                pass  # Ignoring keys that are not a feature's name

        return x


class Prefixer(base.Transformer):
    """Prepends a prefix on features names.

    Parameters
    ----------
    prefix

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12}
    >>> compose.Prefixer('prefix_').transform_one(x)
    {'prefix_a': 42, 'prefix_b': 12}

    """

    def __init__(self, prefix: str):
        self.prefix = prefix

    def _rename(self, s: str) -> str:
        return f"{self.prefix}{s}"

    def transform_one(self, x):
        return {self._rename(i): xi for i, xi in x.items()}


class Suffixer(base.Transformer):
    """Appends a suffix on features names.

    Parameters
    ----------
    suffix

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12}
    >>> compose.Suffixer('_suffix').transform_one(x)
    {'a_suffix': 42, 'b_suffix': 12}

    """

    def __init__(self, suffix: str):
        self.suffix = suffix

    def _rename(self, s: str) -> str:
        return f"{s}{self.suffix}"

    def transform_one(self, x):
        return {self._rename(i): xi for i, xi in x.items()}
