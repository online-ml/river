from river import stream

# from . import base
from river.datasets import base
from .base import ChangePointDataset


class Apple(ChangePointDataset):

    def __init__(self):
        super().__init__(
            annotations={
		"6": [
			319
		],
		"7": [
			319
		],
		"8": [
			319
		],
		"9": [
			53,
			90,
			197,
			276,
			319,
			403,
			463,
			535
		],
		"10": [
			319
		]
	},
            filename="apple.csv",
            task=base.REG,
            n_samples=1867,
            n_features=6,
        )
        self._path = "./datasets/apple.csv"

    def __iter__(self):
        return stream.iter_csv(
            self._path,  # TODO: Must be changed for integration into river
            target=["Open","High","Low","Close","Adj Close","Volume"],
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
