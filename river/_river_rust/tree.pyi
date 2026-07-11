from _typeshed import Incomplete

def log_sum_2_exp(a: float, b: float) -> float: ...
def update_ranges(
    range_min: dict[Incomplete, Incomplete],
    range_max: dict[Incomplete, Incomplete],
    x: dict[Incomplete, Incomplete],
) -> None: ...
def range_extension(
    range_min: dict[Incomplete, Incomplete],
    range_max: dict[Incomplete, Incomplete],
    x: dict[Incomplete, Incomplete],
) -> tuple[float, dict[Incomplete, Incomplete]]: ...
def predict_scores(
    counts: list[Incomplete],
    n_counts: int,
    n_classes: int,
    dirichlet: float,
    n_samples: float,
) -> list[Incomplete]: ...
def go_downwards_classifier(
    root: Incomplete,
    x: dict[Incomplete, Incomplete],
    y_idx: int,
    n_classes: int,
    dirichlet: float,
    use_aggregation: bool,
    step: float,
    split_pure: bool,
    iteration: int,
    max_nodes: int,
    n_nodes: int,
    rng_random: Incomplete,
    rng_choices: Incomplete,
    rng_uniform: Incomplete,
    split_fn: Incomplete,
) -> tuple[Incomplete, Incomplete | None, int]: ...
def go_downwards_regressor(
    root: Incomplete,
    x: dict[Incomplete, Incomplete],
    sample_value: float,
    use_aggregation: bool,
    step: float,
    iteration: int,
    max_nodes: int,
    n_nodes: int,
    rng_random: Incomplete,
    rng_choices: Incomplete,
    rng_uniform: Incomplete,
    split_fn: Incomplete,
) -> tuple[Incomplete, Incomplete | None, int]: ...
def go_upwards(leaf: Incomplete, iteration: int) -> None: ...
def predict_proba_upward(
    leaf: Incomplete, n_classes: int, dirichlet: float
) -> list[Incomplete]: ...
def predict_proba_classifier(
    root: Incomplete,
    x: dict[Incomplete, Incomplete],
    n_classes: int,
    dirichlet: float,
    use_aggregation: bool,
) -> list[Incomplete]: ...
def predict_one_regressor(
    root: Incomplete,
    x: dict[Incomplete, Incomplete],
    use_aggregation: bool,
) -> None: ...
