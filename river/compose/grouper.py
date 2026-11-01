from __future__ import annotations

import collections
import copy
import functools
from typing import Any

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
        transformer: base.typing.Transformer,
        by: base.typing.FeatureName | list[base.typing.FeatureName],
    ):
        self.transformer = transformer
        self.by = by if isinstance(by, list) else [by]
        self.transformers: collections.defaultdict[str, base.typing.Transformer] = (
            collections.defaultdict(functools.partial(copy.deepcopy, transformer))
        )

    def _get_key(self, x: dict[base.typing.FeatureName, Any]) -> str:
        return "_".join(str(x[k]) for k in self.by)

    @property
    def _supervised(self) -> bool:
        return self.transformer._supervised

    def learn_one(
        self,
        x: dict[base.typing.FeatureName, Any],
        y: base.typing.Target | None = None,
    ) -> None:
        key = self._get_key(x)
        transformer = self.transformers[key]
        learn = getattr(transformer, "learn_one", None)
        if learn is None:
            return
        if transformer._supervised:
            learn(x, y)
        else:
            learn(x)

    def transform_one(
        self, x: dict[base.typing.FeatureName, Any]
    ) -> dict[base.typing.FeatureName, Any]:
        key = self._get_key(x)
        return self.transformers[key].transform_one(x)
