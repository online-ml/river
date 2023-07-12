from river import stream

# from . import base
from river.datasets import base
from .base import ChangePointDataset


class UKCoalEmploy(ChangePointDataset):
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
                "6": [
                    15,
                    28,
                    45,
                    60,
                    68,
                    80
                ],
                "7": [
                    18,
                    47,
                    81
                ],
                "8": [],
                "9": [
                    15,
                    27,
                    46,
                    68,
                    81
                ],
                "13": [
                    19,
                    28,
                    45,
                    68,
                    80
                ]
            },
            filename="uk_coal_employment.csv",
            task=base.REG,
            n_samples=105,
            n_features=1,
        )
        self._path = "./datasets/uk_coal_employment.csv"

    def __iter__(self):
        return stream.iter_csv(
            self._path,  # TODO: Must be changed for integration into river
            target="Employment",
            converters={
                "Employment": int,
            },
            parse_dates={"Year": "%Y"},
        )
