from __future__ import annotations

import ast

from river import stream

from . import base


class Restaurants(base.RemoteDataset):
    """Data from the Kaggle Recruit Restaurants challenge.

    The goal is to predict the number of visitors in each of 829 Japanese restaurants over a priod
    of roughly 16 weeks. The data is ordered by date and then by restaurant ID.

    References
    ----------
    [^1]: [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)

    """

    def __init__(self):
        super().__init__(
            n_samples=252_108,
            n_features=7,
            task=base.REG,
            url="https://maxhalford.github.io/files/datasets/kaggle_recruit_restaurants.zip",
            size=28_881_242,
            filename="kaggle_recruit_restaurants.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="visitors",
            converters={
                "latitude": float,
                "longitude": float,
                "visitors": int,
                "is_holiday": ast.literal_eval,
            },
            parse_dates={"date": "%Y-%m-%d"},
        )
