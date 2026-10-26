from typing import Any

def feature_hash(
    x: dict[Any, str | int],
    n_features: int,
    seed: int,
    alternate_sign: bool,
) -> dict[int, int]: ...
