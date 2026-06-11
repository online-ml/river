from __future__ import annotations

import math

import numpy as np
import pandas as pd

from river import datasets, preprocessing, stream


def _pd_split(df, n):
    """Split a pandas DataFrame or Series into n chunks without triggering swapaxes deprecation."""
    indices = np.array_split(range(len(df)), n)
    return [df.iloc[idx] for idx in indices]


def test_standard_scaler_one_many_consistent():
    """Checks that using learn_one or learn_many produces the same result."""
    for with_std in (False, True):
        X = pd.read_csv(datasets.TrumpApproval().path)

        one = preprocessing.StandardScaler(with_std=with_std)
        for x, _ in stream.iter_pandas(X):
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
