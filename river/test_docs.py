"""Test that all public classes and functions have valid docstrings for the doc parser."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import io
import re
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import print_docstring from docs/parse/__main__.py
_spec = importlib.util.spec_from_file_location(
    "docs_parse", REPO_ROOT / "docs" / "parse" / "__main__.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
print_docstring = _mod.print_docstring

# Regex for markdown links: [text](path)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Known broken link patterns (pre-existing issues in the linkifier).
_KNOWN_BROKEN_PATTERNS = [
    "utils/random/Any",
    "api/api/",
    "initializers/Initializer",
    "losses/Loss",
    "../imbalanced-learning",
]


def _iter_public_objects() -> list[tuple[str, Any]]:
    """Return (dotted_name, obj) for every public class and function in river."""
    result: list[tuple[str, Any]] = []
    library = importlib.import_module("river.api")
    for mod_name, mod in inspect.getmembers(library, inspect.ismodule):
        if mod_name.startswith("_"):
            continue
        _iter_module(mod_name, mod, result)
    return result


def _iter_module(prefix: str, mod: types.ModuleType, result: list[tuple[str, Any]]) -> None:
    if not hasattr(mod, "__all__"):
        return
    all_names: list[str] = mod.__all__
    for name, obj in inspect.getmembers(mod):
        if name not in all_names or name.startswith("_"):
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            result.append((f"{prefix}.{name}", obj))
    for name, submod in inspect.getmembers(mod, inspect.ismodule):
        if (
            name in all_names
            and not name.startswith("_")
            and name not in ("tags", "typing", "inspect", "skmultiflow_utils")
        ):
            _iter_module(f"{prefix}.{name}", submod, result)


def _is_broken_link(source_file: Path, target: str) -> bool:
    """Check if a relative markdown link points to a missing file or directory."""
    if target.startswith(("http://", "https://", "#", "mailto:", "/")):
        return False
    target = target.split("#")[0]
    if not target:
        return False
    resolved = (source_file.parent / target).resolve()
    return not (resolved.is_dir() or resolved.is_file() or resolved.with_suffix(".md").is_file())


def test_print_docstring() -> None:
    """Every public object's docstring must be parseable by the doc generator."""
    failures = []
    for name, obj in _iter_public_objects():
        try:
            print_docstring(obj=obj, file=io.StringIO())
        except Exception as exc:
            failures.append(f"  {name}: {exc}")
    if failures:
        pytest.fail("Docstring parsing failed for:\n" + "\n".join(failures))


def test_api_md_has_titles() -> None:
    """Every generated API markdown file must start with an H1 title."""
    api_dir = REPO_ROOT / "docs" / "api"
    if not api_dir.exists():
        pytest.skip("docs/api not built yet")
    missing = []
    for md_file in api_dir.rglob("*.md"):
        if not md_file.read_text().startswith("# "):
            missing.append(str(md_file.relative_to(REPO_ROOT)))
    if missing:
        pytest.fail("Missing H1 title in:\n" + "\n".join(missing))


def test_linkified_internal_links_resolve() -> None:
    """Every internal link in linkified markdown must point to an existing file or directory."""
    linkified = REPO_ROOT / "docs" / "linkified"
    if not linkified.exists():
        pytest.skip("docs/linkified not built yet")
    broken = []
    for md_file in sorted(linkified.rglob("*.md")):
        for match in _LINK_RE.finditer(md_file.read_text()):
            link_text, target = match.group(1), match.group(2)
            if _is_broken_link(md_file, target):
                if not any(pat in target for pat in _KNOWN_BROKEN_PATTERNS):
                    rel = md_file.relative_to(REPO_ROOT)
                    broken.append(f"  {rel}: [{link_text}]({target})")
    if broken:
        pytest.fail("Broken internal links:\n" + "\n".join(broken))
