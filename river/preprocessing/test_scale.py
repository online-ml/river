from __future__ import annotations

import math
import pickle
import random
import typing

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from river import datasets, preprocessing, stats, stream, utils

if typing.TYPE_CHECKING:
    from river.conftest import FrameBackend


def _pd_split(df, n):
    """Split a pandas DataFrame or Series into n chunks without triggering swapaxes deprecation."""
    indices = np.array_split(range(len(df)), n)
    return [df.iloc[idx] for idx in indices]


def test_standard_scaler_one_many_consistent():
    """Checks that using learn_one or learn_many produces the same result."""
    for with_std in (False, True):
        X = pd.read_csv(datasets.TrumpApproval().path)

        one = preprocessing.StandardScaler(with_std=with_std)
        for x, _ in stream.iter_frame(X):
            one.learn_one(x)

        many = preprocessing.StandardScaler(with_std=with_std)
        for xb in _pd_split(X, 10):
            many.learn_many(xb)

        for i in X:
            assert math.isclose(one.counts[i], many.counts[i])
            assert math.isclose(one.means[i], many.means[i])
            assert math.isclose(one.vars[i], many.vars[i])


def test_standard_scaler_shuffle_columns():
    """Checks that learn_many works identically whether columns are shuffled or not."""
    X = pd.read_csv(datasets.TrumpApproval().path)

    normal = preprocessing.StandardScaler()
    for xb in _pd_split(X, 10):
        normal.learn_many(xb)

    shuffled = preprocessing.StandardScaler()
    for xb in _pd_split(X, 10):
        cols = np.random.permutation(X.columns)
        shuffled.learn_many(xb[cols])

    for i in X:
        assert math.isclose(shuffled.counts[i], shuffled.counts[i])
        assert math.isclose(shuffled.means[i], shuffled.means[i])
        assert math.isclose(shuffled.vars[i], shuffled.vars[i])


def test_standard_scaler_add_remove_columns():
    """Checks that no exceptions are raised whenever columns are dropped and/or added."""
    X = pd.read_csv(datasets.TrumpApproval().path)

    ss = preprocessing.StandardScaler()
    for xb in _pd_split(X, 10):
        # Pick half of the columns at random
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        ss.learn_many(xb[cols])


def test_minmax_scaler_warm_start():
    """`_from_state` seeds min/max so the very first transform uses them."""
    scaler = preprocessing.MinMaxScaler._from_state(min={"x": 8.0}, max={"x": 12.0})
    assert scaler.transform_one({"x": 10.0}) == {"x": 0.5}
    assert scaler.transform_one({"x": 8.0}) == {"x": 0.0}
    assert scaler.transform_one({"x": 12.0}) == {"x": 1.0}


def test_minmax_scaler_warm_start_extends_after_learn():
    """A subsequent learn_one with a value below the seeded min lowers the running min."""
    scaler = preprocessing.MinMaxScaler._from_state(min={"x": 8.0}, max={"x": 12.0})
    scaler.learn_one({"x": 6.0})
    assert scaler.min["x"].get() == 6.0
    assert scaler.max["x"].get() == 12.0
    assert scaler.transform_one({"x": 6.0}) == {"x": 0.0}


def test_minmax_scaler_warm_start_partial_keys():
    """Features absent from the warm-start dicts get fresh default stats."""
    scaler = preprocessing.MinMaxScaler._from_state(min={"x": 1.0}, max={"x": 5.0})
    scaler.learn_one({"x": 3.0, "y": 7.0})
    assert scaler.transform_one({"x": 3.0, "y": 7.0}) == {"x": 0.5, "y": 0.0}


def test_minmax_scaler_window_size_uses_rolling_stats():
    """With window_size set, the scaler relies on RollingMin/RollingMax."""
    from river import stats

    scaler = preprocessing.MinMaxScaler(window_size=3)
    for v in [10.0, 20.0, 30.0, 40.0]:
        scaler.learn_one({"x": v})
    assert isinstance(scaler.min["x"], stats.RollingMin)
    assert isinstance(scaler.max["x"], stats.RollingMax)
    assert scaler.min["x"].get() == 20.0
    assert scaler.max["x"].get() == 40.0


def test_minmax_scaler_window_evicts_old_values():
    """Values beyond the window no longer influence the scaling range."""
    scaler = preprocessing.MinMaxScaler(window_size=2)
    for v in [1.0, 100.0, 5.0, 10.0]:
        scaler.learn_one({"x": v})
    assert scaler.transform_one({"x": 5.0}) == {"x": 0.0}
    assert scaler.transform_one({"x": 10.0}) == {"x": 1.0}


def test_minmax_scaler_default_unchanged():
    """Zero-arg construction must remain behavior-identical to the previous version."""
    scaler = preprocessing.MinMaxScaler()
    assert scaler.window_size is None
    for v in [10.0, 20.0, 30.0]:
        scaler.learn_one({"x": v})
    assert scaler.transform_one({"x": 20.0}) == {"x": 0.5}


def test_maxabs_scaler_warm_start():
    scaler = preprocessing.MaxAbsScaler._from_state(abs_max={"x": 12.0})
    assert scaler.transform_one({"x": 6.0}) == {"x": 0.5}
    assert scaler.transform_one({"x": -12.0}) == {"x": -1.0}


def test_maxabs_scaler_window_evicts_old_values():
    scaler = preprocessing.MaxAbsScaler(window_size=2)
    for v in [-100.0, 1.0, 2.0]:
        scaler.learn_one({"x": v})
    assert scaler.transform_one({"x": 2.0}) == {"x": 1.0}


def test_standard_scaler_warm_start():
    """`_from_state` seeds counts/means/vars so transform uses them immediately."""
    scaler = preprocessing.StandardScaler._from_state(
        counts={"x": 100},
        means={"x": 10.0},
        vars={"x": 4.0},
    )
    assert scaler.transform_one({"x": 12.0}) == {"x": 1.0}
    assert scaler.transform_one({"x": 10.0}) == {"x": 0.0}


def test_standard_scaler_warm_start_then_extends():
    """After warm-start, learn_one should keep refining the running statistics."""
    scaler = preprocessing.StandardScaler._from_state(
        counts={"x": 1},
        means={"x": 10.0},
        vars={"x": 0.0},
    )
    scaler.learn_one({"x": 20.0})
    assert scaler.counts["x"] == 2
    assert math.isclose(scaler.means["x"], 15.0)


def test_minmax_scaler_from_state_with_window_size():
    """`_from_state` with `window_size` set seeds the rolling window."""
    from river import stats

    scaler = preprocessing.MinMaxScaler._from_state(
        min={"x": 0.0},
        max={"x": 100.0},
        window_size=3,
    )
    assert isinstance(scaler.min["x"], stats.RollingMin)
    assert scaler.transform_one({"x": 50.0}) == {"x": 0.5}
    # Seed value gets evicted after window_size more observations.
    for v in [10.0, 20.0, 30.0]:
        scaler.learn_one({"x": v})
    assert scaler.min["x"].get() == 10.0
    assert scaler.max["x"].get() == 30.0


def test_maxabs_scaler_from_state_partial_keys():
    """Features absent from the warm-start dict get fresh default stats."""
    scaler = preprocessing.MaxAbsScaler._from_state(abs_max={"x": 4.0})
    scaler.learn_one({"x": 2.0, "y": 3.0})
    out = scaler.transform_one({"x": 2.0, "y": 3.0})
    assert out == {"x": 0.5, "y": 1.0}


def test_minmax_scaler_clone_preserves_window_size():
    """`clone()` reconstructs window_size via __init__ but drops learned state."""
    from river import stats

    scaler = preprocessing.MinMaxScaler(window_size=5)
    scaler.learn_one({"x": 1.0})
    cloned = scaler.clone()
    assert cloned.window_size == 5
    assert isinstance(cloned.min["x"], stats.RollingMin)
    # Learned state must not carry over.
    assert len(cloned.min["x"].window) == 0


def test_maxabs_scaler_clone_preserves_window_size():
    scaler = preprocessing.MaxAbsScaler(window_size=7)
    cloned = scaler.clone()
    assert cloned.window_size == 7


def test_standard_scaler_warm_start_matches_running():
    """A warm-started scaler should match one trained from scratch on the same data."""
    data = [{"x": v} for v in [1.0, 2.0, 3.0, 4.0, 5.0]]

    full = preprocessing.StandardScaler()
    for x in data:
        full.learn_one(x)

    # Take a snapshot after the first 3 observations and resume from it.
    snapshot = preprocessing.StandardScaler()
    for x in data[:3]:
        snapshot.learn_one(x)

    resumed = preprocessing.StandardScaler._from_state(
        counts=dict(snapshot.counts),
        means=dict(snapshot.means),
        vars=dict(snapshot.vars),
    )
    for x in data[3:]:
        resumed.learn_one(x)

    assert resumed.counts["x"] == full.counts["x"]
    assert math.isclose(resumed.means["x"], full.means["x"])
    assert math.isclose(resumed.vars["x"], full.vars["x"])


def test_standard_scaler_warm_start_then_learn_many():
    """Warm-start must be compatible with the mini-batch `learn_many` path."""
    snapshot = preprocessing.StandardScaler._from_state(
        counts={"x": 2},
        means={"x": 1.5},
        vars={"x": 0.25},
    )
    snapshot.learn_many(pd.DataFrame({"x": [2.0, 3.0]}))
    assert snapshot.counts["x"] == 4
    assert math.isclose(snapshot.means["x"], 2.0)


def test_standard_scaler_warm_start_without_std():
    """When with_std is False the vars argument is ignored, not required."""
    scaler = preprocessing.StandardScaler._from_state(
        counts={"x": 5},
        means={"x": 3.0},
        with_std=False,
    )
    assert scaler.transform_one({"x": 5.0}) == {"x": 2.0}


def _np_population_stats(values: list[float]) -> tuple[float, float]:
    """Compute population mean and variance (ddof=0) for a list of floats."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.var(ddof=0))


def test_standard_scaler_window_default_unchanged():
    """Zero-arg construction stays behavior-identical to the previous version."""
    scaler = preprocessing.StandardScaler()
    assert scaler.window_size is None
    assert isinstance(scaler.means["x"], float)
    assert isinstance(scaler.vars["x"], float)


def test_standard_scaler_window_uses_rolling_stats():
    """With window_size set, per-feature means/vars are Rolling wrappers."""
    scaler = preprocessing.StandardScaler(window_size=3)
    scaler.learn_one({"x": 1.0})
    assert isinstance(scaler.means["x"], utils.Rolling)
    assert isinstance(scaler.vars["x"], utils.Rolling)
    assert isinstance(scaler.means["x"].obj, stats.Mean)
    assert isinstance(scaler.vars["x"].obj, stats.Var)
    # ddof=0 to match the Welford population-variance estimator.
    assert scaler.vars["x"].obj.ddof == 0


def test_standard_scaler_window_independent_per_feature():
    """Each feature must have its own underlying Mean/Var instance.

    This guards against the silent-aliasing bug that occurs when the
    `defaultdict` factory shares a single stats instance across keys.
    """
    scaler = preprocessing.StandardScaler(window_size=10)
    scaler.learn_one({"x": 1.0, "y": 100.0})
    scaler.learn_one({"x": 2.0, "y": 200.0})

    assert scaler.means["x"] is not scaler.means["y"]
    assert scaler.means["x"].obj is not scaler.means["y"].obj
    assert scaler.vars["x"].obj is not scaler.vars["y"].obj
    assert math.isclose(scaler.means["x"].get(), 1.5)
    assert math.isclose(scaler.means["y"].get(), 150.0)


def test_standard_scaler_window_matches_numpy_pop_var():
    """The rolling mean/var must equal a NumPy ddof=0 computation over the window."""
    rng = random.Random(0)
    window_size = 5
    scaler = preprocessing.StandardScaler(window_size=window_size)
    values = [rng.uniform(-10, 10) for _ in range(25)]

    for i, v in enumerate(values, start=1):
        scaler.learn_one({"x": v})
        window = values[max(0, i - window_size) : i]
        expected_mean, expected_var = _np_population_stats(window)
        assert math.isclose(scaler.means["x"].get(), expected_mean, abs_tol=1e-9)
        assert math.isclose(scaler.vars["x"].get(), expected_var, abs_tol=1e-9)


def test_standard_scaler_window_transform_one_matches_numpy():
    """transform_one must compute (x - rolling_mean) / rolling_std."""
    rng = random.Random(42)
    window_size = 4
    scaler = preprocessing.StandardScaler(window_size=window_size)
    values = [rng.uniform(0, 50) for _ in range(20)]

    for i, v in enumerate(values, start=1):
        scaler.learn_one({"x": v})
        out = scaler.transform_one({"x": v})
        window = values[max(0, i - window_size) : i]
        m, var = _np_population_stats(window)
        expected = (v - m) / var**0.5 if var else 0.0
        assert math.isclose(out["x"], expected, abs_tol=1e-9)


def test_standard_scaler_window_equivalent_below_threshold():
    """While n <= window_size, the windowed and running scalers must agree exactly."""
    rng = random.Random(7)
    window_size = 8
    running = preprocessing.StandardScaler()
    windowed = preprocessing.StandardScaler(window_size=window_size)

    for _ in range(window_size):
        x = {"x": rng.uniform(-5, 5)}
        running.learn_one(x)
        windowed.learn_one(x)
        r_out = running.transform_one(x)
        w_out = windowed.transform_one(x)
        assert math.isclose(r_out["x"], w_out["x"], abs_tol=1e-9)


def test_standard_scaler_window_adapts_to_drift():
    """After a drift, the windowed scaler's mean must move toward the new regime
    faster than the global running scaler.
    """
    pre_drift = [1.0] * 50
    post_drift = [100.0] * 10
    running = preprocessing.StandardScaler()
    windowed = preprocessing.StandardScaler(window_size=5)
    for v in pre_drift + post_drift:
        x = {"x": v}
        running.learn_one(x)
        windowed.learn_one(x)
    assert windowed.means["x"].get() == 100.0
    assert running.means["x"] < 50.0


def test_standard_scaler_window_with_std_false():
    """`with_std=False` skips the variance update path under windowed mode."""
    scaler = preprocessing.StandardScaler(with_std=False, window_size=3)
    for v in [1.0, 2.0, 3.0, 4.0]:
        scaler.learn_one({"x": v})
    # Mean is updated; vars stay at their default empty Rolling(Var).
    assert math.isclose(scaler.means["x"].get(), 3.0)
    out = scaler.transform_one({"x": 4.0})
    assert math.isclose(out["x"], 4.0 - 3.0)


def test_standard_scaler_window_learn_many_matches_learn_one():
    """In windowed mode, learn_many should be equivalent to row-by-row learn_one."""
    rng = random.Random(1)
    values = [rng.uniform(-5, 5) for _ in range(30)]
    df = pd.DataFrame({"x": values, "y": [v * 2 for v in values]})

    one = preprocessing.StandardScaler(window_size=7)
    for _, row in df.iterrows():
        one.learn_one(row.to_dict())

    many = preprocessing.StandardScaler(window_size=7)
    many.learn_many(df)

    assert math.isclose(one.means["x"].get(), many.means["x"].get(), abs_tol=1e-9)
    assert math.isclose(one.vars["x"].get(), many.vars["x"].get(), abs_tol=1e-9)
    assert math.isclose(one.means["y"].get(), many.means["y"].get(), abs_tol=1e-9)
    assert math.isclose(one.vars["y"].get(), many.vars["y"].get(), abs_tol=1e-9)


def test_standard_scaler_window_learn_many_chunks_equivalent():
    """Splitting a stream into mini-batches must not change the final rolling state."""
    rng = random.Random(2)
    df = pd.DataFrame({"x": [rng.uniform(-3, 3) for _ in range(50)]})

    monolithic = preprocessing.StandardScaler(window_size=10)
    monolithic.learn_many(df)

    chunked = preprocessing.StandardScaler(window_size=10)
    for piece in _pd_split(df, 7):
        chunked.learn_many(piece)

    assert math.isclose(monolithic.means["x"].get(), chunked.means["x"].get(), abs_tol=1e-9)
    assert math.isclose(monolithic.vars["x"].get(), chunked.vars["x"].get(), abs_tol=1e-9)


def test_standard_scaler_window_transform_many_matches_transform_one():
    """transform_many under windowed mode must match per-row transform_one."""
    rng = random.Random(3)
    df = pd.DataFrame({"x": [rng.uniform(-2, 2) for _ in range(15)]})

    scaler = preprocessing.StandardScaler(window_size=4)
    scaler.learn_many(df)

    out_many = scaler.transform_many(df)
    out_one = pd.DataFrame([scaler.transform_one(row.to_dict()) for _, row in df.iterrows()])
    np.testing.assert_allclose(out_many.values, out_one.values, atol=1e-9)


def test_standard_scaler_window_clone_preserves_window_size():
    """`clone()` reconstructs window_size via __init__ but drops learned state."""
    scaler = preprocessing.StandardScaler(window_size=5, with_std=False)
    scaler.learn_one({"x": 1.0})
    cloned = scaler.clone()
    assert cloned.window_size == 5
    assert cloned.with_std is False
    assert isinstance(cloned.means["x"], utils.Rolling)
    assert len(cloned.means["x"].window) == 0


def test_standard_scaler_window_pickle_roundtrip():
    """A windowed scaler must survive pickle and keep producing correct outputs."""
    scaler = preprocessing.StandardScaler(window_size=4)
    for v in [1.0, 2.0, 3.0, 4.0]:
        scaler.learn_one({"x": v})

    restored = pickle.loads(pickle.dumps(scaler))
    assert restored.window_size == 4
    assert isinstance(restored.means["x"], utils.Rolling)
    assert math.isclose(restored.means["x"].get(), scaler.means["x"].get())
    assert math.isclose(restored.vars["x"].get(), scaler.vars["x"].get())

    # Continued learning after unpickling stays correct (no aliased state).
    restored.learn_one({"x": 5.0})
    expected_mean = (2.0 + 3.0 + 4.0 + 5.0) / 4
    assert math.isclose(restored.means["x"].get(), expected_mean)


def test_standard_scaler_window_pipeline_integration():
    """A windowed scaler composes through `|` like any other Transformer."""
    from river import compose

    pipeline = compose.Select("x") | preprocessing.StandardScaler(window_size=3)
    for v in [1.0, 2.0, 3.0]:
        pipeline.learn_one({"x": v, "ignored": 99.0})
    out = pipeline.transform_one({"x": 3.0, "ignored": 0.0})
    # mean(1,2,3)=2, popvar=2/3, std=sqrt(2/3); (3-2)/std
    assert math.isclose(out["x"], (3.0 - 2.0) / (2.0 / 3.0) ** 0.5)


def test_standard_scaler_window_counts_track_cumulative_observations():
    """In windowed mode, counts still track the cumulative observation count."""
    scaler = preprocessing.StandardScaler(window_size=3)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        scaler.learn_one({"x": v})
    assert scaler.counts["x"] == 5


def _strip_attr(obj, attr):
    """Simulate an old pickle by deleting an attribute added later."""
    new = obj.__class__.__new__(obj.__class__)
    new.__dict__.update({k: v for k, v in obj.__dict__.items() if k != attr})
    return new


def test_standard_scaler_unpickle_missing_window_size():
    """Old pickles (pre-`window_size`) must keep working after upgrade."""
    scaler = preprocessing.StandardScaler()
    for v in [1.0, 2.0, 3.0]:
        scaler.learn_one({"x": v})

    legacy = pickle.loads(pickle.dumps(_strip_attr(scaler, "window_size")))
    assert legacy.window_size is None
    legacy.learn_one({"x": 4.0})
    out = legacy.transform_one({"x": 4.0})
    assert isinstance(out["x"], float)


def test_minmax_scaler_unpickle_missing_window_size():
    scaler = preprocessing.MinMaxScaler()
    for v in [1.0, 2.0, 3.0]:
        scaler.learn_one({"x": v})

    legacy = pickle.loads(pickle.dumps(_strip_attr(scaler, "window_size")))
    assert legacy.window_size is None
    legacy.learn_one({"x": 4.0})
    assert legacy.transform_one({"x": 4.0}) == {"x": 1.0}


def test_maxabs_scaler_unpickle_missing_window_size():
    scaler = preprocessing.MaxAbsScaler()
    for v in [1.0, 2.0, 3.0]:
        scaler.learn_one({"x": v})

    legacy = pickle.loads(pickle.dumps(_strip_attr(scaler, "window_size")))
    assert legacy.window_size is None
    legacy.learn_one({"x": 4.0})
    assert legacy.transform_one({"x": 4.0}) == {"x": 1.0}


def test_issue_1313():
    """>>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from river import preprocessing
    >>> from river.compose import Select

    >>> X, y = datasets.make_regression(n_samples=6, n_features=2)
    >>> X = pd.DataFrame(X)
    >>> X.columns = ['feat_1','feat_2']

    >>> model = Select('feat_1') | preprocessing.StandardScaler()
    >>> X = X.astype('float32')
    >>> X.dtypes
    feat_1    float32
    feat_2    float32
    dtype: object

    >>> model.learn_many(X)
    >>> X1 = model.transform_many(X)
    >>> X1.dtypes
    feat_1    float32
    dtype: object

    """


# `StandardScaler`'s mini-batch path is routed through narwhals: every backend takes the same
# backend-agnostic path, which infers the compute dtype from the input schema and so preserves the
# input's float dtype. These tests pin the cross-backend behaviour, using the pandas path as the
# oracle.


def _numeric_batch(n: int = 40) -> dict[str, list[float]]:
    """Build a reproducible batch of two numeric columns."""
    rng = random.Random(42)
    return {
        "x": [rng.uniform(8, 12) for _ in range(n)],
        "y": [rng.uniform(-5, 5) for _ in range(n)],
    }


def _trump_columns() -> dict[str, list[float]]:
    """The TrumpApproval dataset as a column dict (mixed int64/float64 features)."""
    return pd.read_csv(datasets.TrumpApproval().path).to_dict(orient="list")


def _chunk_dict(data: dict[str, list], n: int) -> list[dict[str, list]]:
    """Split a column dict into `n` row-wise chunks, preserving column order."""
    length = len(next(iter(data.values())))
    bounds = np.array_split(range(length), n)
    return [{col: [values[i] for i in idx] for col, values in data.items()} for idx in bounds]


# Mini-batches with a shifting column set: ``a`` disappears then reappears, ``c`` emerges, and the
# column order is permuted between calls. `StandardScaler` advertises support for this, so it must
# behave identically on every backend.
EMERGING_BATCHES: list[dict[str, list[float]]] = [
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]},
    {"c": [-1.0, -2.0, -3.0], "b": [50.0, 60.0, 70.0]},
    {"b": [80.0, 90.0], "a": [5.0, 6.0], "c": [-4.0, -5.0]},
]
# A frame whose columns were all seen during the batches above, used to probe `transform_many`.
EMERGING_TRANSFORM: dict[str, list[float]] = {
    "a": [1.5, 6.5, 3.0],
    "b": [15.0, 95.0, 45.0],
    "c": [-1.5, -4.5, -2.0],
}


def _frame_to_numpy(native) -> np.ndarray:
    """Read a native frame back into a float64 numpy matrix via narwhals."""
    return np.asarray(nw.from_native(native, eager_only=True).to_numpy(), dtype=np.float64)


@pytest.mark.parametrize("with_std", [True, False])
@pytest.mark.parametrize("window_size", [None, 7])
def test_standard_scaler_transform_many_backend_agnostic(
    frame_backend: FrameBackend, with_std: bool, window_size: int | None
) -> None:
    """`transform_many` yields the same values on every backend as the pandas reference."""
    data = _numeric_batch()

    reference = preprocessing.StandardScaler(with_std=with_std, window_size=window_size)
    reference.learn_many(pd.DataFrame(data))
    expected = reference.transform_many(pd.DataFrame(data)).to_numpy()

    scaler = preprocessing.StandardScaler(with_std=with_std, window_size=window_size)
    scaler.learn_many(frame_backend.frame(data))
    got = _frame_to_numpy(scaler.transform_many(frame_backend.frame(data)))

    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("window_size", [None, 7])
def test_standard_scaler_learn_many_backend_agnostic(
    frame_backend: FrameBackend, window_size: int | None
) -> None:
    """`learn_many` accumulates identical running statistics across backends."""
    data = _numeric_batch()

    reference = preprocessing.StandardScaler(window_size=window_size)
    reference.learn_many(pd.DataFrame(data))

    scaler = preprocessing.StandardScaler(window_size=window_size)
    scaler.learn_many(frame_backend.frame(data))

    for col in data:
        ref_mean = reference.means[col].get() if window_size else reference.means[col]
        got_mean = scaler.means[col].get() if window_size else scaler.means[col]
        ref_var = reference.vars[col].get() if window_size else reference.vars[col]
        got_var = scaler.vars[col].get() if window_size else scaler.vars[col]
        assert scaler.counts[col] == reference.counts[col]
        assert math.isclose(got_mean, ref_mean, abs_tol=1e-9)
        assert math.isclose(got_var, ref_var, abs_tol=1e-9)


def test_standard_scaler_transform_many_returns_native_backend(
    frame_backend: FrameBackend,
) -> None:
    """The output frame is rebuilt in the caller's own backend."""
    data = _numeric_batch()
    native = frame_backend.frame(data)

    scaler = preprocessing.StandardScaler()
    scaler.learn_many(native)
    out = scaler.transform_many(native)

    assert type(out) is type(native)


def test_standard_scaler_transform_many_chunked_backend_agnostic(
    frame_backend: FrameBackend,
) -> None:
    """Learning in mini-batches then transforming matches the pandas reference."""
    data = _numeric_batch(60)
    pdf = pd.DataFrame(data)

    reference = preprocessing.StandardScaler()
    for chunk in _pd_split(pdf, 5):
        reference.learn_many(chunk)
    expected = reference.transform_many(pdf).to_numpy()

    scaler = preprocessing.StandardScaler()
    native = frame_backend.frame(data)
    # Slice the native frame into the same row ranges via narwhals.
    nw_frame = nw.from_native(native, eager_only=True)
    bounds = np.array_split(range(len(pdf)), 5)
    for idx in bounds:
        scaler.learn_many(nw_frame[int(idx[0]) : int(idx[-1]) + 1].to_native())
    got = _frame_to_numpy(scaler.transform_many(native))

    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_real_dataset_backend_agnostic(
    frame_backend: FrameBackend, with_std: bool
) -> None:
    """A real, mixed-dtype dataset (TrumpApproval) scales identically on every backend.

    The frame mixes an ``int64`` column with several ``float64`` ones and is learned in chunks,
    exercising both the running-statistics merge and the integer-to-float promotion path.
    """
    data = _trump_columns()
    chunks = _chunk_dict(data, 8)

    reference = preprocessing.StandardScaler(with_std=with_std)
    for chunk in chunks:
        reference.learn_many(pd.DataFrame(chunk))
    expected = reference.transform_many(pd.DataFrame(data)).to_numpy()

    scaler = preprocessing.StandardScaler(with_std=with_std)
    for chunk in chunks:
        scaler.learn_many(frame_backend.frame(chunk))
    got = _frame_to_numpy(scaler.transform_many(frame_backend.frame(data)))

    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("with_std", [True, False])
@pytest.mark.parametrize("window_size", [None, 5])
def test_standard_scaler_emerging_features_learn_backend_agnostic(
    frame_backend: FrameBackend, with_std: bool, window_size: int | None
) -> None:
    """Adding/removing/reordering columns between `learn_many` calls is backend-agnostic.

    Each feature's running statistics must depend only on the rows in which it actually appears,
    identically across backends, even as the column set and order change between mini-batches.
    """
    reference = preprocessing.StandardScaler(with_std=with_std, window_size=window_size)
    scaler = preprocessing.StandardScaler(with_std=with_std, window_size=window_size)
    for batch in EMERGING_BATCHES:
        reference.learn_many(pd.DataFrame(batch))
        scaler.learn_many(frame_backend.frame(batch))

    for col in ("a", "b", "c"):
        ref_mean = reference.means[col].get() if window_size else reference.means[col]
        got_mean = scaler.means[col].get() if window_size else scaler.means[col]
        assert scaler.counts[col] == reference.counts[col]
        assert math.isclose(got_mean, ref_mean, abs_tol=1e-9)
        if with_std:
            ref_var = reference.vars[col].get() if window_size else reference.vars[col]
            got_var = scaler.vars[col].get() if window_size else scaler.vars[col]
            assert math.isclose(got_var, ref_var, abs_tol=1e-9)


def test_standard_scaler_emerging_features_transform_backend_agnostic(
    frame_backend: FrameBackend,
) -> None:
    """After learning on a shifting column set, `transform_many` matches the pandas oracle."""
    reference = preprocessing.StandardScaler()
    scaler = preprocessing.StandardScaler()
    for batch in EMERGING_BATCHES:
        reference.learn_many(pd.DataFrame(batch))
        scaler.learn_many(frame_backend.frame(batch))

    expected = reference.transform_many(pd.DataFrame(EMERGING_TRANSFORM)).to_numpy()
    got = _frame_to_numpy(scaler.transform_many(frame_backend.frame(EMERGING_TRANSFORM)))

    np.testing.assert_allclose(got, expected, atol=1e-9)
