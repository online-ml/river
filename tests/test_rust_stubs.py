"""Keep the hand-written stubs for `river._river_rust.stats` in sync with the compiled module."""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import ModuleType

import river
from river._river_rust import drift as rust_drift
from river._river_rust import feature_hashing as rust_feature_hashing
from river._river_rust import stats as rust_stats
from river._river_rust import tree as rust_tree
from river._river_rust import vectordict as rust_vectordict

# Dunders that PyO3 defines on the classes and that the stubs are expected to declare.
PICKLE_DUNDERS = {"__getstate__", "__setstate__", "__getnewargs__"}


def stub_api(stub_filename: str) -> dict[str, set[str] | None]:
    """Return the stub's API: class name -> method names, function name -> None."""
    stub_path = pathlib.Path(river.__file__).parent / "_river_rust" / stub_filename
    api: dict[str, set[str] | None] = {}
    for node in ast.parse(stub_path.read_text()).body:
        if isinstance(node, ast.ClassDef):
            api[node.name] = {
                stmt.name
                for stmt in node.body
                if isinstance(stmt, ast.FunctionDef)
                and (not stmt.name.startswith("__") or stmt.name in PICKLE_DUNDERS)
            } - {"__init__"}
        elif isinstance(node, ast.FunctionDef):
            api[node.name] = None
    return api


def runtime_api(rust_module: ModuleType) -> dict[str, set[str] | None]:
    """Return the compiled module's API in the same shape as `stub_api`."""
    api: dict[str, set[str] | None] = {}
    for name, obj in vars(rust_module).items():
        if name.startswith("__"):
            continue
        if inspect.isclass(obj):
            api[name] = {
                attr
                for attr, member in vars(obj).items()
                # Properties are not considered callables but getset descriptors
                # so we need to handle them specifically
                if (callable(member) or inspect.isgetsetdescriptor(member))
                and (not attr.startswith("__") or attr in PICKLE_DUNDERS)
            }
        else:
            api[name] = None
    return api


def test_stubs_match_runtime() -> None:
    for stub_filename, rust_module in [
        ("stats.pyi", rust_stats),
        ("drift.pyi", rust_drift),
        ("tree.pyi", rust_tree),
        ("vectordict.pyi", rust_vectordict),
        ("feature_hashing.pyi", rust_feature_hashing),
    ]:
        stub = stub_api(stub_filename)
        runtime = runtime_api(rust_module)
        assert stub.keys() == runtime.keys()
        for name, runtime_members in runtime.items():
            stub_members = stub[name]
            assert (stub_members is None) == (runtime_members is None), name
            if stub_members is not None:
                assert stub_members == runtime_members, name
