from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class Occupancy(ChangePointFileDataset):
    """Room occupancy data.

    Dataset on detecting room occupancy based on several variables. The dataset contains
    temperature, humidity, light, and CO2 variables.

    The data is sampled at every 16 observations to reduce the length of the series.

    References
    ----------
    Candanedo, Luis M., and Véronique Feldheim. "Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models." Energy and Buildings 112 (2016): 28-39.

    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [238, 416],
                "8": [53, 143, 238, 417],
                "9": [53, 92, 142, 181, 236, 264, 341, 416, 436, 451, 506],
                "10": [1, 52, 91, 142, 181, 234, 267, 324, 360, 416, 451, 506],
                "12": [234, 415],
            },
            filename="occupancy.csv",
            task=datasets.base.REG,
            n_samples=509,
            n_features=4,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=["V1", "V2", "V3", "V4"],
            converters={
                "V1": float,
                "V2": float,
                "V3": float,
                "V4": float,
            },
            parse_dates={"time": "%Y-%m-%d %H:%M:%S"},
        )
