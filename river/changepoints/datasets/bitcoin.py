from river import stream

# from . import base
from river.datasets import base
from .base import ChangePointDataset


class Bitcoin(ChangePointDataset):
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
                "6": [
                    502,
                    580,
                    702,
                    747
                ],
                "8": [
                    583
                ],
                "12": [
                    597
                ],
                "13": [
                    522,
                    579,
                    591,
                    629,
                    703,
                    747,
                    760
                ],
                "14": [
                    93,
                    522,
                    540,
                    701,
                    747,
                    760,
                    772
                ]
            },
            filename="bitcoin.csv",
            task=base.REG,
            n_samples=822,
            n_features=1,
        )
        self._path = "./datasets/bitcoin.csv"

    def __iter__(self):
        return stream.iter_csv(
            self._path,  # TODO: Must be changed for integration into river
            target="price",
            converters={
                "price": float,
            },
            parse_dates={"date": "%Y-%m-%d"},
        )
