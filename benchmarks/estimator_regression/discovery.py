"""Estimator discovery and inventory audit.

Discovery mirrors ``river/test_estimators.py``: walk ``river.api.__all__``,
import each submodule, and collect concrete ``base.Estimator`` subclasses.

Caveats handled here (verified against the codebase):

* ``river.api.__all__`` includes ``base`` itself, and ``base.Estimator`` is not
  abstract by ``inspect.isabstract``. Discovery skips the ``base`` submodule so
  the root ``Estimator`` class is never counted as a concrete estimator.
* ``inspect.isabstract`` filters interface classes such as ``base.Classifier``.
* Only classes whose defining module is the discovered submodule are kept, so
  re-exports do not double-count estimators across modules.

River-to-sklearn wrappers (``River2SKL*``) are sklearn estimators, not
``base.Estimator`` subclasses, so the ``issubclass(obj, base.Estimator)``
check already excludes them — no special-casing is needed.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from river import base


@dataclass(frozen=True)
class DiscoveredEstimator:
    """A concrete public estimator class discovered from the River API."""

    module: str
    class_name: str
    qualname: str  # Defining module path, e.g. "river.linear_model.log_reg.LogisticRegression".

    @property
    def id(self) -> str:
        """Stable public identifier used by estimator regression scenarios."""

        return f"{self.module}.{self.class_name}"


def _is_concrete_estimator(obj: Any, submodule: str) -> bool:
    if not inspect.isclass(obj):
        return False
    if not issubclass(obj, base.Estimator):
        return False
    if inspect.isabstract(obj):
        return False
    module_path_parts = obj.__module__.split(".")
    if len(module_path_parts) <= 1:
        return False
    return module_path_parts[1] == submodule


def discover() -> list[DiscoveredEstimator]:
    """Return every concrete public estimator, sorted by module then class."""

    api = importlib.import_module("river.api")
    found: list[DiscoveredEstimator] = []
    for submodule in api.__all__:
        if submodule == "base":
            continue
        mod = importlib.import_module(f"river.{submodule}")
        for _, obj in inspect.getmembers(
            mod, predicate=lambda o: _is_concrete_estimator(o, submodule)
        ):
            found.append(
                DiscoveredEstimator(
                    module=submodule,
                    class_name=obj.__name__,
                    qualname=f"{obj.__module__}.{obj.__name__}",
                )
            )
    return sorted(found, key=lambda e: (e.module, e.class_name))
