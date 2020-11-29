import collections
import copy
import functools
import typing

from river import base


__all__ = ["Grouper"]


class Grouper(base.Transformer):
    """Applies a transformer within different groups.

    This transformer allows you to split your data into groups and apply a transformer within each
    group. This happens in a streaming manner, which means that the groups are discovered online.
    A separate copy of the provided transformer is made whenever a new group appears. The groups
    are defined according to one or more keys.

    Parameters
    ----------
    transformer
    by
        The field on which to group the data. This can either by a single value, or a list of
        values.

    """

    def __init__(
        self,
        transformer: base.Transformer,
        by: typing.Union[base.typing.FeatureName, typing.List[base.typing.FeatureName]],
    ):

        self.transformer = transformer
        self.by = by if isinstance(by, list) else [by]
        self.transformers = collections.defaultdict(functools.partial(copy.deepcopy, transformer))

    def _get_key(self, x):
        return "_".join(str(x[k]) for k in self.by)

    def learn_one(self, x):
        key = self._get_key(x)
        self.transformers[key].learn_one(x)
        return self

    def transform_one(self, x):
        key = self._get_key(x)
        return self.transformers[key].transform_one(x)
