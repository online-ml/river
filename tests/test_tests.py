"""Test that the tests test what they are suppposed to test, because clearly that was needed."""

from __future__ import annotations

from conftest import _ROOT, _uses_pandas


def test_pandas_detection():
    """Every test that loads `pandas`, even indirectly, should be detected as so.

    On the other hand, files that don't need pandas shouldn't trigger false positives.
    This is necessary to achieve the pandas/not pandas test split in CI.
    """
    assert not _uses_pandas(_ROOT / "tests/test_docs.py")
    assert not _uses_pandas(_ROOT / "tests/time_series/test_holt_winters.py")
    assert not _uses_pandas(_ROOT / "tests/test_estimators.py")
    assert _uses_pandas(_ROOT / "tests/test_estimators_pandas.py")
    assert _uses_pandas(_ROOT / "tests/utils/test_pandas.py")
    assert _uses_pandas(_ROOT / "tests/stream/test_iter_frame.py")
    assert _uses_pandas(_ROOT / "river/checks/common.py")
    assert _uses_pandas(_ROOT / "tests/preprocessing/test_one_hot.py")
    assert _uses_pandas(_ROOT / "tests/preprocessing/test_ordinal.py")
    assert _uses_pandas(_ROOT / "tests/utils/test_dataframe.py")
    assert _uses_pandas(_ROOT / "tests/anomaly/test_lof.py")
    assert _uses_pandas(_ROOT / "tests/frames.py")
    assert not _uses_pandas(
        _ROOT / "tests/tree/test_native_hoeffding_tree.py"
    )  # false negative exception
