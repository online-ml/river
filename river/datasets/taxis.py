from __future__ import annotations

from river import stream

from . import base


class Taxis(base.RemoteDataset):
    """Taxi ride durations in New York City.

    The goal is to predict the duration of taxi rides in New York City.

    References
    ----------
    [^1]: [New York City Taxi Trip Duration competition on Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration)

    """

    def __init__(self):
        super().__init__(
            n_samples=1_458_644,
            n_features=8,
            task=base.REG,
            url="https://maxhalford.github.io/files/datasets/nyc_taxis.zip",
            size=195_271_696,
            filename="train.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="trip_duration",
            converters={
                "passenger_count": int,
                "pickup_longitude": float,
                "pickup_latitude": float,
                "dropoff_longitude": float,
                "dropoff_latitude": float,
                "trip_duration": int,
            },
            parse_dates={"pickup_datetime": "%Y-%m-%d %H:%M:%S"},
            drop=["dropoff_datetime", "id"],
        )
