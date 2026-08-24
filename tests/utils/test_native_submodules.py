from __future__ import annotations

import importlib


def test_native_submodules_resolve_by_dotted_path() -> None:
    for name in ("stats", "drift", "tree", "vectordict"):
        mod = importlib.import_module(f"river._river_rust.{name}")
        assert mod is not None, f"river._river_rust.{name} did not import"
