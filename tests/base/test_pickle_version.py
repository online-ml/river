from __future__ import annotations

import pickle
import sys
import warnings

import pytest

from river import base, exceptions, linear_model, misc, preprocessing, stats
from river.__version__ import __version__
from river.base import base as base_module


class SlottedBase(base.Base):
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 3


def test_slotted_subclass_roundtrip() -> None:
    restored = pickle.loads(pickle.dumps(SlottedBase()))

    assert restored.value == 3


def test_pickle_records_version_without_changing_model() -> None:
    model = linear_model.LinearRegression()
    model.learn_one({"x": 2}, 4)

    state = model.__getstate__()

    assert state["_river_version"] == __version__
    assert "_river_version" not in model.__dict__

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(model))

    assert not recorded
    assert restored.predict_one({"x": 2}) == model.predict_one({"x": 2})
    assert "_river_version" not in restored.__dict__


def test_pickle_warns_on_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = stats.Mean()
    model.update(3)
    with monkeypatch.context() as context:
        context.setattr(base_module, "__version__", "0.0.0")
        payload = pickle.dumps(model)

    with pytest.warns(exceptions.InconsistentVersionWarning) as recorded:
        restored = pickle.loads(payload)

    warning = recorded[0].message
    assert isinstance(warning, exceptions.InconsistentVersionWarning)
    assert warning.estimator_name == "Mean"
    assert warning.original_river_version == "0.0.0"
    assert warning.current_river_version == __version__
    assert restored.get() == model.get()
    assert "_river_version" not in restored.__dict__


def test_pipeline_can_continue_after_version_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    for x, y in [({"x": 1}, 2), ({"x": 2}, 4), ({"x": 3}, 6)]:
        model.learn_one(x, y)
    expected = model.predict_one({"x": 4})

    with monkeypatch.context() as context:
        context.setattr(base_module, "__version__", "0.0.0")
        payload = pickle.dumps(model)

    with pytest.warns(exceptions.InconsistentVersionWarning) as recorded:
        restored = pickle.loads(payload)

    names = set()
    for warning in recorded:
        message = warning.message
        assert isinstance(message, exceptions.InconsistentVersionWarning)
        names.add(message.estimator_name)
    assert names >= {
        "Pipeline",
        "StandardScaler",
        "LinearRegression",
    }
    assert restored.predict_one({"x": 4}) == expected
    restored.learn_one({"x": 4}, 8)


@pytest.mark.parametrize(
    "model_class",
    [
        stats.Mean,
        preprocessing.StandardScaler,
        preprocessing.MinMaxScaler,
        preprocessing.MaxAbsScaler,
        pytest.param(
            misc.ZstdClassifier,
            marks=pytest.mark.skipif(sys.version_info < (3, 14), reason="requires Python 3.14"),
        ),
    ],
)
def test_legacy_pickle_warns_on_unknown_version(
    model_class: type[base.Base], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = model_class()
    with monkeypatch.context() as context:
        context.setattr(base.Base, "__getstate__", lambda self: self.__dict__.copy())
        payload = pickle.dumps(model)

    with pytest.warns(exceptions.InconsistentVersionWarning) as recorded:
        restored = pickle.loads(payload)

    warning = recorded[0].message
    assert isinstance(warning, exceptions.InconsistentVersionWarning)
    assert warning.original_river_version == "unknown"
    assert type(restored) is type(model)


@pytest.mark.parametrize(
    "model_class",
    [
        preprocessing.StandardScaler,
        preprocessing.MinMaxScaler,
        preprocessing.MaxAbsScaler,
        pytest.param(
            misc.ZstdClassifier,
            marks=pytest.mark.skipif(sys.version_info < (3, 14), reason="requires Python 3.14"),
        ),
    ],
)
def test_custom_pickle_hooks_warn_on_version_mismatch(
    model_class: type[base.Base], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = model_class()
    with monkeypatch.context() as context:
        context.setattr(base_module, "__version__", "0.0.0")
        payload = pickle.dumps(model)

    with pytest.warns(exceptions.InconsistentVersionWarning) as recorded:
        restored = pickle.loads(payload)

    warning = recorded[0].message
    assert isinstance(warning, exceptions.InconsistentVersionWarning)
    assert warning.original_river_version == "0.0.0"
    assert type(restored) is type(model)
