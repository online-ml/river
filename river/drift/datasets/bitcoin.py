from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class Bitcoin(ChangePointFileDataset):
    """Bitcoin Market Price

    This is a regression task, where the goal is to predict the average USD market price across
    major bitcoin exchanges. This data was collected from the official Blockchain website. There
    is only one feature given, the day of exchange, which is in increments of three. The first
    500 lines have been removed because they are not interesting.

    References
    ----------
    [^1]: https://www.blockchain.com/fr/explorer/charts/market-price?timespan=all

    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [502, 580, 702, 747],
                "8": [583],
                "12": [597],
                "13": [522, 579, 591, 629, 703, 747, 760],
                "14": [93, 522, 540, 701, 747, 760, 772],
            },
            filename="bitcoin.csv",
            task=datasets.base.REG,
            n_samples=822,
            n_features=1,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="price",
            converters={
                "price": float,
            },
            parse_dates={"date": "%Y-%m-%d"},
        )
