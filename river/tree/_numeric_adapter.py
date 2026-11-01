from __future__ import annotations

from typing import Any, cast

from river import base


class NumericAdapter:
    def __init__(
        self,
        x: dict[base.typing.FeatureName, Any],
        y: base.typing.ClfTarget,
        sort_features: bool = False,
    ):
        self.feature_names = list(sorted(cast(Any, x)) if sort_features else x)
        self.features = {feature: index for index, feature in enumerate(self.feature_names)}
        self.labels = {y: 0}
        self.classes = [y]
        self.output_labels = [y]
        self.output_indices = [0]

    def add_label(self, y: base.typing.ClfTarget) -> tuple[int, bool]:
        try:
            return self.labels[y], False
        except KeyError:
            index = len(self.classes)
            self.classes.append(y)
            self.labels[y] = index
            self.output_labels = sorted(self.classes)
            self.output_indices = [self.labels[label] for label in self.output_labels]
            return index, True
