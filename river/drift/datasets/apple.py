from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class Apple(ChangePointFileDataset):
    """Apple Stock

    This dataset concerns the daily close price and volume of Apple stock around the year 2000. The dataset is sampled every 3 observations to reduce the length of the time series.
    This dataset is retrieved from Yahoo Finance.

    References
    ----------
    [^1]: https://finance.yahoo.com/quote/AAPL/history?period1=850348800&period2=1084579200&interval=1d&filter=history&frequency=1d

    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [319],
                "7": [319],
                "8": [319],
                "9": [53, 90, 197, 276, 319, 403, 463, 535],
                "10": [319],
            },
            filename="apple.csv",
            task=datasets.base.REG,
            n_samples=1867,
            n_features=6,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            converters={
                "Open": float,
                "High": float,
                "Low": float,
                "Close": float,
                "Adj Close": float,
                "Volume": float,
            },
            parse_dates={"Date": "%Y-%m-%d"},
        )
