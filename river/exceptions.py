from __future__ import annotations


class InconsistentVersionWarning(UserWarning):
    def __init__(
        self,
        *,
        estimator_name: str,
        current_river_version: str,
        original_river_version: str,
    ) -> None:
        self.estimator_name = estimator_name
        self.current_river_version = current_river_version
        self.original_river_version = original_river_version

    def __str__(self) -> str:
        return (
            f"Trying to unpickle {self.estimator_name} from River version "
            f"{self.original_river_version} while using version {self.current_river_version}. "
            "This might lead to incompatible behavior."
        )
