from collections.abc import Callable, Hashable, Sequence
from random import Random
from typing import Any

from river.base.typing import FeatureName
from river.tree.mondrian.mondrian_tree_nodes import MondrianLeafClassifier, MondrianLeafRegressor

class NativeHoeffdingTreeClassifier:
    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        max_features: int | None = None,
        grace_period: float = 200.0,
        delta: float = 1e-7,
        tau: float = 0.05,
        n_splits: int = 10,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        leaf_prediction: str = "nba",
        nb_threshold: float = 0.0,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        merit_preprune: bool = True,
        seed: int = 42,
    ) -> None: ...
    def predict_one(self, x: list[float]) -> int: ...
    def predict_proba_one(self, x: list[float]) -> list[float]: ...
    def predict_proba_mapping(
        self,
        x: dict[FeatureName, Any],
        feature_indices: dict[FeatureName, int],
        output_labels: list[Any],
        output_indices: list[int],
    ) -> dict[Any, float]: ...
    def predict_mapping(
        self,
        x: dict[FeatureName, Any],
        feature_indices: dict[FeatureName, int],
        class_order: list[int],
    ) -> int: ...
    def learn_one(self, x: list[float], y: int, w: float = 1.0) -> None: ...
    def learn_one_mapping(
        self,
        x: dict[FeatureName, Any],
        feature_indices: dict[FeatureName, int],
        feature_names: list[FeatureName],
        y: int,
        w: float = 1.0,
    ) -> None: ...
    def learn_one_with_rng(self, x: list[float], y: int, rng: Random, w: float = 1.0) -> None: ...
    def learn_one_mapping_with_rng(
        self,
        x: dict[FeatureName, Any],
        feature_indices: dict[FeatureName, int],
        feature_names: list[FeatureName],
        y: int,
        rng: Random,
        w: float = 1.0,
    ) -> None: ...
    def n_nodes(self) -> int: ...
    def n_branches(self) -> int: ...
    def n_leaves(self) -> int: ...
    def n_active_leaves(self) -> int: ...
    def height(self) -> int: ...
    def total_weight_seen(self) -> float: ...
    def nodes(self) -> list[tuple[int, float, int, int, list[float], int, bool]]: ...
    def add_features(self, count: int) -> None: ...
    def add_class(self) -> None: ...
    def insert_class(self, index: int, grow: bool) -> None: ...
    def set_mutable_parameters(self, grace_period: float, delta: float, tau: float) -> None: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
    def __getnewargs__(self) -> tuple[int, int]: ...

def log_sum_2_exp(a: float, b: float) -> float: ...
def update_ranges(
    range_min: dict[Hashable, float],
    range_max: dict[Hashable, float],
    x: dict[Hashable, float],
) -> None: ...
def range_extension(
    range_min: dict[Hashable, float],
    range_max: dict[Hashable, float],
    x: dict[Hashable, float],
) -> tuple[float, dict[Hashable, float]]: ...
def predict_scores(
    counts: list[float], n_counts: int, n_classes: int, dirichlet: float, n_samples: int
) -> list[float]: ...
def go_downwards_classifier(
    root: MondrianLeafClassifier | object,
    x: dict[Hashable, float],
    y_idx: int,
    n_classes: int,
    dirichlet: float,
    use_aggregation: bool,
    step: float,
    split_pure: bool,
    iteration: int,
    max_nodes: int,
    n_nodes: int,
    rng_random: Callable[[], float],
    rng_choices: Callable[[Sequence[object], Sequence[float] | None], list[object]],
    rng_uniform: Callable[[float, float], float],
    split_fn: Callable[[MondrianLeafClassifier, float, float, FeatureName, bool], object],
) -> tuple[MondrianLeafClassifier, MondrianLeafClassifier | None, int]: ...
def go_downwards_regressor(
    root: object,
    x: dict[Hashable, float],
    sample_value: float,
    use_aggregation: bool,
    step: float,
    iteration: int,
    max_nodes: int,
    n_nodes: int,
    rng_random: Callable[[], float],
    rng_choices: Callable[[Sequence[object], Sequence[float] | None], list[object]],
    rng_uniform: Callable[[float, float], float],
    split_fn: Callable[[MondrianLeafRegressor, float, float, FeatureName, bool], object],
) -> tuple[MondrianLeafRegressor, MondrianLeafRegressor | None, int]: ...
def go_upwards(leaf: MondrianLeafClassifier | MondrianLeafRegressor, iteration: int) -> None: ...
def predict_proba_upward(leaf: Any, n_classes: int, dirichlet: float) -> list[float]: ...
def predict_proba_classifier(
    root: MondrianLeafClassifier,
    x: dict[Hashable, float],
    n_classes: int,
    dirichlet: float,
    use_aggregation: bool,
) -> list[float]: ...
def predict_one_regressor(
    root: MondrianLeafRegressor, x: dict[Hashable, float], use_aggregation: bool
) -> float: ...
