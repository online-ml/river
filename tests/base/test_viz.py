from __future__ import annotations

import re

from river.base import viz


def _css_rule(selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", viz.CSS, re.S)
    assert match is not None
    return match.group("body")


def test_union_elements_are_displayed_vertically() -> None:
    assert "flex-direction: column;" in _css_rule(".river-union")

    union_spacing = _css_rule(".river-union > .river-component + .river-component")
    assert "margin-top: 1em;" in union_spacing
    assert "margin-left" not in union_spacing
