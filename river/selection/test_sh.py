import pytest

from river import selection, linear_model, metrics
from river.selection.exceptions import NotEnoughModels


def test_sh_only_one_model():
    with pytest.raises(NotEnoughModels):
        selection.SuccessiveHalvingClassifier(
            [linear_model.LogisticRegression()], metric=metrics.LogLoss(), budget=42
        )
    with pytest.raises(NotEnoughModels):
        selection.SuccessiveHalvingRegressor(
            [linear_model.LinearRegression()], metric=metrics.MAE(), budget=42
        )
