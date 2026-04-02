"""Test that all public classes and functions have valid docstrings for the doc parser."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import io
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import print_docstring from docs/parse/__main__.py
_spec = importlib.util.spec_from_file_location(
    "docs_parse", REPO_ROOT / "docs" / "parse" / "__main__.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
print_docstring = _mod.print_docstring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_public_objects():
    """Yield (dotted_name, obj) for every public class and function in river."""
    library = importlib.import_module("river.api")
    for mod_name, mod in inspect.getmembers(library, inspect.ismodule):
        if mod_name.startswith("_"):
            continue
        yield from _iter_module(mod_name, mod)


def _iter_module(prefix, mod):
    if not hasattr(mod, "__all__"):
        return
    ispublic = lambda x: x.__name__ in mod.__all__ and not x.__name__.startswith("_")
    for name, obj in inspect.getmembers(
        mod, lambda x: (inspect.isclass(x) or inspect.isfunction(x)) and ispublic(x)
    ):
        yield f"{prefix}.{name}", obj
    for name, submod in inspect.getmembers(mod, inspect.ismodule):
        if (
            name in mod.__all__
            and not name.startswith("_")
            and name not in ("tags", "typing", "inspect")
        ):
            yield from _iter_module(f"{prefix}.{name}", submod)


PUBLIC_OBJECTS = list(_iter_public_objects())

# Regex for markdown links: [text](path)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Known broken link patterns (pre-existing issues in the linkifier).
# The linkifier auto-links dotted names it finds in text, which can
# produce links to non-existent pages (base classes, type annotations,
# deprecated functions) or with wrong relative prefixes.
_KNOWN_BROKEN_PATTERNS = [
    "utils/random/Any",  # type annotation falsely linkified
    "api/api/",  # double prefix from non-__all__ members
    "initializers/Initializer",  # base class, no public API page
    "losses/Loss",  # base class, no public API page
    "../imbalanced-learning",  # relative link in example notebook, valid on the site
]


def _resolve_link(source_file: Path, target: str) -> Path | None:
    """Resolve a relative markdown link to an absolute path in the docs tree.

    Returns the resolved Path if it can be found (as a file, directory, or
    .md file), or None if the link is external / anchor-only / absolute.
    """
    # Skip external URLs, anchors, and site-root-relative links
    if target.startswith(("http://", "https://", "#", "mailto:", "/")):
        return None

    # Strip fragment
    target = target.split("#")[0]
    if not target:
        return None

    resolved = (source_file.parent / target).resolve()

    # It might point to a directory (which has an index.md) or a .md file
    if resolved.is_dir() or resolved.is_file():
        return resolved
    if resolved.with_suffix(".md").is_file():
        return resolved.with_suffix(".md")

    return resolved  # return the unresolved path so the test can report it


def _is_known_broken(target: str) -> bool:
    """Check if a link target matches a known pre-existing broken pattern."""
    return any(pat in target for pat in _KNOWN_BROKEN_PATTERNS)


def _collect_linkified_md_files():
    """Collect all .md files in docs/linkified/ (if they exist)."""
    linkified = REPO_ROOT / "docs" / "linkified"
    if not linkified.exists():
        return []
    return sorted(linkified.rglob("*.md"))


# ---------------------------------------------------------------------------
# Tests: docstring parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,obj", PUBLIC_OBJECTS, ids=[name for name, _ in PUBLIC_OBJECTS])
def test_print_docstring(name, obj):
    """Every public object's docstring must be parseable by the doc generator."""
    buf = io.StringIO()
    print_docstring(obj=obj, file=buf)


# ---------------------------------------------------------------------------
# Tests: generated API markdown content
# ---------------------------------------------------------------------------


def _collect_api_md_files():
    """Collect generated API markdown files."""
    api_dir = REPO_ROOT / "docs" / "api"
    if not api_dir.exists():
        return []
    return sorted(api_dir.rglob("*.md"))


API_MD_FILES = _collect_api_md_files()


@pytest.mark.skipif(not API_MD_FILES, reason="docs/api not built yet")
@pytest.mark.parametrize(
    "md_file", API_MD_FILES, ids=[str(f.relative_to(REPO_ROOT)) for f in API_MD_FILES]
)
def test_api_md_has_title(md_file):
    """Every generated API markdown file must start with an H1 title."""
    text = md_file.read_text()
    assert text.startswith("# "), f"{md_file.name} does not start with an H1 heading"


@pytest.mark.skipif(not API_MD_FILES, reason="docs/api not built yet")
@pytest.mark.parametrize(
    "md_file", API_MD_FILES, ids=[str(f.relative_to(REPO_ROOT)) for f in API_MD_FILES]
)
def test_api_md_no_none_content(md_file):
    """Generated markdown must not contain stringified None values in headings or parameter types."""
    text = md_file.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("#") and line.strip() == "# None":
            pytest.fail(f"{md_file.name}:{i} has 'None' as a heading")
        if line.strip() == "*Type* → *None*":
            pytest.fail(f"{md_file.name}:{i} has 'None' as a type annotation")


# ---------------------------------------------------------------------------
# Tests: link validation in linkified docs
# ---------------------------------------------------------------------------


LINKIFIED_MD_FILES = _collect_linkified_md_files()


@pytest.mark.skipif(not LINKIFIED_MD_FILES, reason="docs/linkified not built yet")
@pytest.mark.parametrize(
    "md_file",
    LINKIFIED_MD_FILES,
    ids=[str(f.relative_to(REPO_ROOT)) for f in LINKIFIED_MD_FILES],
)
def test_linkified_internal_links_resolve(md_file):
    """Every internal link in linkified markdown must point to an existing file or directory."""
    text = md_file.read_text()
    broken = []

    for match in _LINK_RE.finditer(text):
        link_text, target = match.group(1), match.group(2)

        resolved = _resolve_link(md_file, target)
        if resolved is None:
            continue  # external or anchor

        if not (resolved.is_dir() or resolved.is_file()):
            if not _is_known_broken(target):
                broken.append(f"  [{link_text}]({target}) -> {resolved}")

    if broken:
        rel = md_file.relative_to(REPO_ROOT)
        details = "\n".join(broken)
        pytest.fail(f"{rel} has broken internal links:\n{details}")
