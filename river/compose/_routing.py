from __future__ import annotations

import inspect

from river import base


def method_params(step: base.Estimator, method: str) -> frozenset[str] | None:
    try:
        params = inspect.signature(getattr(type(step), method)).parameters
    except (ValueError, TypeError, AttributeError):
        return None
    if any(param.kind is param.VAR_KEYWORD for param in params.values()):
        return None
    return frozenset(params) - {"self", "x", "y", "X"}


def route(accepted: frozenset[str] | None, params: dict[str, object]) -> dict[str, object]:
    if accepted is None or not params:
        return params
    return {name: params[name] for name in accepted & params.keys()}
