from __future__ import annotations

from river import datasets, stream

from .base import ChangePointFileDataset


class RunLog(ChangePointFileDataset):
    """Interval Training Running Pace.

    This dataset shows the pace of a runner during an interval training session, where a mobile
    application provides instructions on when to run and when to walk.

    """

    def __init__(self):
        super().__init__(
            annotations={
                "6": [60, 96, 114, 174, 204, 240, 258, 317],
                "7": [60, 96, 114, 177, 204, 240, 258, 317],
                "8": [60, 96, 114, 174, 204, 240, 258, 317],
                "10": [2, 60, 96, 114, 174, 204, 240, 258, 317],
                "12": [],
            },
            filename="run_log.csv",
            task=datasets.base.REG,
            n_samples=376,
            n_features=2,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=["Pace", "Distance"],
            converters={"Pace": float, "Distance": float},
            parse_dates={"time": "%Y-%m-%d %H:%M:%S"},
        )
