from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class BrentSpotPrice(ChangePointFileDataset):
    """Brent Spot Price

    This is the USD price for Brent Crude oil, measured daily. We include the time series
        from 2000 onwards. The data is sampled at every 10 original observations to reduce the length of the series.

        The data is obtained from the U.S. Energy Information Administration. Since the data is in the public domain,
        we distribute it as part of this repository.

        Since the original data has observations only on trading days, there are arguably gaps in this time
        series (on non-trading days). However we consider these to be consecutive, and thus also consider
        the sampled time series to have consecutive observations.

    References
    ----------
    [^1]: U.S. Energy Information Administration (Sep. 2019)
    [^2]: https://www.eia.gov/opendata/v1/qb.php?sdid=PET.RBRTE.D

    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [219, 230, 288],
                "8": [227, 381],
                "9": [86, 219, 230, 279, 375],
                "12": [169, 172, 217, 228, 287, 368, 382, 389, 409],
                "13": [170, 180, 219, 229, 246, 271, 286, 379, 409, 444, 483],
            },
            filename="brent_crude_oil.csv",
            task=datasets.base.REG,
            n_samples=8_195,
            n_features=1,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="DPB",
            converters={
                "DPB": float,
            },
            parse_dates={"day": "%m/%d/%Y"},
        )
