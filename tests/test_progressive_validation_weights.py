"""
Tests for per-sample weight support in _progressive_validation.

The module is loaded via importlib with river's Rust/Cython dependencies
pre-mocked, so these tests run without a compiled build.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Pre-mock river's Rust-dependent submodules BEFORE any river import fires.
# This file lives outside the `river` package so pytest imports it standalone,
# giving our sys.modules patches time to land before progressive_validation.py
# is exec'd via importlib below.
# ---------------------------------------------------------------------------
_mock_metrics = MagicMock()
_mock_metrics.base.Metrics = type("Metrics", (), {})

for _name, _mock in {
    "river": MagicMock(),
    "river.base": MagicMock(),
    "river.metrics": _mock_metrics,
    "river.metrics.base": _mock_metrics.base,
    "river.stream": MagicMock(),
    "river.utils": MagicMock(),
}.items():
    sys.modules.setdefault(_name, _mock)

# Load the source file directly so CI tests the local branch, not an install.
_pv_path = Path(__file__).parent.parent / "river" / "evaluate" / "progressive_validation.py"
_spec = importlib.util.spec_from_file_location("_pv", _pv_path)
_pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pv)  # type: ignore[union-attr]
_progressive_validation = _pv._progressive_validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(accepts_w: bool = True):
    import inspect

    model = MagicMock()
    model._supervised = True
    model.predict_one = MagicMock(return_value=0)
    model._raw_memory_usage = 0

    if accepts_w:
        def _learn(x, y, w=1.0):
            pass
        model.learn_one = MagicMock()
        model.learn_one.__signature__ = inspect.signature(_learn)
    else:
        def _learn_no_w(x, y):
            pass
        model.learn_one = MagicMock()
        model.learn_one.__signature__ = inspect.signature(_learn_no_w)

    return model


def _make_metric():
    metric = MagicMock()
    metric.works_with.return_value = True
    metric.requires_labels = True
    return metric


def _fake_simulate_qa(ds, moment, delay, copy):
    """No-delay simulation: question then answer for each sample."""
    for i, (x, y) in enumerate(ds):
        yield i, x, None  # question
        yield i, x, y     # answer


def _run(dataset, accepts_w=True):
    """Run _progressive_validation and return the w values passed to learn_one."""
    model = _make_model(accepts_w=accepts_w)
    metric = _make_metric()

    mock_utils = MagicMock()
    mock_utils.inspect.isanomalydetector.return_value = False
    mock_utils.inspect.isanomalyfilter.return_value = False
    mock_utils.inspect.isclassifier.return_value = False
    mock_utils.inspect.isactivelearner.return_value = False

    mock_stream = MagicMock()
    mock_stream.simulate_qa.side_effect = _fake_simulate_qa

    _pv.utils = mock_utils
    _pv.stream = mock_stream

    list(
        _progressive_validation(
            dataset=dataset,
            model=model,
            metric=metric,
            checkpoints=iter([]),
        )
    )

    return [c.kwargs.get("w") for c in model.learn_one.call_args_list]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerSampleWeights:
    def test_xy_pairs_default_to_weight_one(self):
        """(x, y) pairs — learn_one must receive w=1.0 for every sample."""
        dataset = [
            ({"f": 1}, 0),
            ({"f": 2}, 1),
            ({"f": 3}, 0),
        ]
        assert _run(dataset) == [1.0, 1.0, 1.0]

    def test_xyw_triples_forward_per_sample_weight(self):
        """(x, y, w) triples — each sample's w must reach learn_one."""
        dataset = [
            ({"f": 1}, 0, 2.0),
            ({"f": 2}, 1, 0.5),
            ({"f": 3}, 0, 3.0),
        ]
        assert _run(dataset) == [2.0, 0.5, 3.0]

    def test_mixed_tuples_default_missing_weight(self):
        """(x, y) and (x, y, w) mixed — missing w defaults to 1.0."""
        dataset = [
            ({"f": 1}, 0),
            ({"f": 2}, 1, 4.0),
            ({"f": 3}, 0),
        ]
        assert _run(dataset) == [1.0, 4.0, 1.0]

    def test_model_without_w_never_receives_w_kwarg(self):
        """Models whose learn_one has no w param must not receive w."""
        dataset = [
            ({"f": 1}, 0, 2.0),
            ({"f": 2}, 1, 0.5),
        ]
        model = _make_model(accepts_w=False)
        metric = _make_metric()

        mock_utils = MagicMock()
        mock_utils.inspect.isanomalydetector.return_value = False
        mock_utils.inspect.isanomalyfilter.return_value = False
        mock_utils.inspect.isclassifier.return_value = False
        mock_utils.inspect.isactivelearner.return_value = False

        mock_stream = MagicMock()
        mock_stream.simulate_qa.side_effect = _fake_simulate_qa

        _pv.utils = mock_utils
        _pv.stream = mock_stream

        list(
            _progressive_validation(
                dataset=dataset,
                model=model,
                metric=metric,
                checkpoints=iter([]),
            )
        )

        for c in model.learn_one.call_args_list:
            assert "w" not in c.kwargs
            assert len(c.args) == 2  # only (x, y)
