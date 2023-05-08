from __future__ import annotations

from river import stream

from . import base


class AirlinePassengers(base.FileDataset):
    """Monthly number of international airline passengers.

    The stream contains 144 items and only one single feature, which is the month. The goal is to
    predict the number of passengers each month by capturing the trend and the seasonality of the
    data.

    References
    ----------
    [^1]: [International airline passengers: monthly totals in thousands. Jan 49 – Dec 60](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line)

    """

    def __init__(self):
        super().__init__(
            filename="airline-passengers.csv",
            task=base.REG,
            n_features=1,
            n_samples=144,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="passengers",
            converters={"passengers": int},
            parse_dates={"month": "%Y-%m"},
        )
