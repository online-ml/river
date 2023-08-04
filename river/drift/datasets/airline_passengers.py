from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class AirlinePassengers(ChangePointFileDataset):
    """JFK Airline Passengers

    This dataset gives the number of passengers arriving and departing at JFK.
    The data is obtained from New York State's official Kaggle page for this dataset.

    References
    ----------
    [^1]: https://www.kaggle.com/new-york-state/nys-air-passenger-traffic,-port-authority-of-ny-nj#air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv

    """

    def __init__(self):
        super().__init__(
            annotations={"6": [299], "7": [], "8": [302], "9": [326, 382], "10": [296]},
            filename="airline_passengers.csv",
            task=datasets.base.REG,
            n_samples=468,
            n_features=1,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="Total Passengers",
            converters={
                "Total Passengers": int,
            },
            parse_dates={"date": "%Y-%b"},
        )
