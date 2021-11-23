from river import stream

from . import base


class Bikes(base.RemoteDataset):
    """Bike sharing station information from the city of Toulouse.

    The goal is to predict the number of bikes in 5 different bike stations from the city of
    Toulouse.

    References
    ----------
    [^1]: [A short introduction and conclusion to the OpenBikes 2016 Challenge](https://maxhalford.github.io/blog/openbikes-challenge/)

    """

    def __init__(self):
        super().__init__(
            url="https://maxhalford.github.io/files/datasets/toulouse_bikes.zip",
            size=13_125_015,
            n_samples=182_470,
            n_features=8,
            task=base.REG,
            filename="toulouse_bikes.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="bikes",
            converters={
                "clouds": int,
                "humidity": int,
                "pressure": float,
                "temperature": float,
                "wind": float,
                "bikes": int,
            },
            parse_dates={"moment": "%Y-%m-%d %H:%M:%S"},
        )
