from __future__ import annotations

import typing

import narwhals.stable.v2 as nw
import numpy as np
import pytest

from river import utils

if typing.TYPE_CHECKING:
    from tests.frames import FrameBackend

DATA: dict[str, list[float]] = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}


def test_to_numpy_maps_nulls_to_nan(frame_backend: FrameBackend) -> None:
    """Missing values reach the numpy core as `NaN`, whatever the backend represents them with."""
    frame = frame_backend.frame({"a": [1.5, None, 3.25], "b": [4.5, 5.25, None]})

    values = utils.dataframe.to_numpy(utils.dataframe.into_frame(frame))

    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, np.array([[1.5, 4.5], [np.nan, 5.25], [3.25, np.nan]]))


@pytest.mark.parametrize("dtype", ["double[pyarrow]", "Float64", "float64"])
def test_to_numpy_casts_pandas_extension_floats_with_nulls(dtype: str) -> None:
    """Guards the `pandas[pyarrow]` case above against pandas' dtype inference changing."""

    # NOTE:`frame_backend` reaches these dtypes through `convert_dtypes`, so what it actually produces
    # depends on the data. Here the dtype is pinned instead, keeping the coverage explicit.

    pytest.importorskip("pandas")

    import pandas as pd

    frame = pd.DataFrame({"a": [1.5, None, 3.25], "b": [4.5, 5.25, None]}).astype(
        {"a": dtype, "b": dtype}
    )

    values = utils.dataframe.to_numpy(utils.dataframe.into_frame(frame))

    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, np.array([[1.5, 4.5], [np.nan, 5.25], [3.25, np.nan]]))


def test_to_numpy_honours_requested_dtype(frame_backend: FrameBackend) -> None:
    frame = utils.dataframe.into_frame(frame_backend.frame(DATA))
    values = utils.dataframe.to_numpy(frame, dtype=np.float32)
    assert values.dtype == np.float32


def test_to_numpy_honours_int_dtype(frame_backend: FrameBackend) -> None:
    frame = utils.dataframe.into_frame(frame_backend.frame(DATA))
    values = utils.dataframe.to_numpy(frame, dtype=np.int64)

    assert values.dtype == np.int64
    np.testing.assert_array_equal(values, np.array([[1, 4], [2, 5], [3, 6]]))


def test_to_native_frame_from_array(frame_backend: FrameBackend) -> None:
    """The array fast path rebuilds a frame with the input's backend and column order."""
    frame = utils.dataframe.into_frame(frame_backend.frame(DATA))
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    out = utils.dataframe.into_frame(
        utils.dataframe.to_native_frame(values, like=frame, columns=["u", "v"])
    )

    assert out.implementation == frame.implementation
    assert [str(col) for col in out.columns] == ["u", "v"]
    np.testing.assert_allclose(utils.dataframe.to_numpy(out), values)


def test_to_native_series(frame_backend: FrameBackend) -> None:
    frame = utils.dataframe.into_frame(frame_backend.frame(DATA))

    series = nw.from_native(
        utils.dataframe.to_native_series([7.0, 8.0, 9.0], name="y", like=frame), series_only=True
    )

    assert series.implementation == frame.implementation
    assert series.to_list() == [7.0, 8.0, 9.0]
