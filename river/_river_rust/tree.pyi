from collections.abc import Callable, Hashable, Sequence
from typing import Any

from river.base.typing import FeatureName
from river.tree.mondrian.mondrian_tree_nodes import MondrianLeafClassifier, MondrianLeafRegressor

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
