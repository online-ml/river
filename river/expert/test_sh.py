import pytest

from river import expert, linear_model, metrics
from river.expert.exceptions import NotEnoughModels


def test_sh_only_one_model():
    with pytest.raises(NotEnoughModels):
        expert.SuccessiveHalvingClassifier(
            [linear_model.LogisticRegression()], metric=metrics.LogLoss(), budget=42
        )
    with pytest.raises(NotEnoughModels):
        expert.SuccessiveHalvingRegressor(
            [linear_model.LinearRegression()], metric=metrics.MAE(), budget=42
        )
