from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class UKCoalEmploy(ChangePointFileDataset):
    """Historic Employment in UK Coal Mines

    This is historic data obtained from the UK government.
        We use the employment column for the number of workers employed in the British coal mines
        Missing values in the data are replaced with the value of the preceding year.

    References
    ----------
    [^1]: https://www.gov.uk/government/statistical-data-sets/historical-coal-data-coal-production-availability-and-consumption
    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [15, 28, 45, 60, 68, 80],
                "7": [18, 47, 81],
                "8": [],
                "9": [15, 27, 46, 68, 81],
                "13": [19, 28, 45, 68, 80],
            },
            filename="uk_coal_employment.csv",
            task=datasets.base.REG,
            n_samples=105,
            n_features=1,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="Employment",
            converters={
                "Employment": int,
            },
            parse_dates={"Year": "%Y"},
        )
