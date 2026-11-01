from __future__ import annotations

import pytest

from river import facto


@pytest.mark.parametrize(
    "model",
    [
        facto.FFMRegressor(sample_normalization=True, seed=42),
        facto.FMRegressor(sample_normalization=True, seed=42),
        facto.FwFMRegressor(sample_normalization=True, seed=42),
        facto.HOFMRegressor(sample_normalization=True, seed=42),
    ],
)
def test_sample_normalization_accepts_zero_vector(model):
    x = {"x": 0.0}

    model.learn_one(x, 0.0)

    assert model.predict_one(x) == pytest.approx(0.0)
