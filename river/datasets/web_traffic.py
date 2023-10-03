from __future__ import annotations

import csv

from river import stream

from . import base


class PipeCSVDialect(csv.unix_dialect):
    delimiter = "|"


class WebTraffic(base.RemoteDataset):
    """Web sessions information from an events company based in South Africa.

    The goal is to predict the number of web sessions in 4 different regions in South Africa.

    The data consists of 15 minute interval traffic values between '2023-06-16 00:00:00' and
    '2023-09-15 23:45:00' for each region. Two types of sessions are captured `sessionsA` and
    `sessionsB`. The `isMissing` flag is equal to 1 if any of the servers failed to capture
    sessions, otherwise if all servers functioned properly this flag is equal to 0.

    Things to consider:

    * region `R5` captures sessions in backup mode. Strictly speaking, `R5` is not necessary to predict.
    * Can `sessionsA` and `sessionsB` events be predicted accurately for each region over the next day (next 96 intervals)?
    * What is the best way to deal with the missing values?
    * How can model selection be used (a multi-model approach)?
    * Can dependence (correlation) between regions be utilised for more accurate predictions?
    * Can both `sessionA` and `sessionB` be predicted simultaneously with one model?

    This dataset is well suited for time series forecasting models, as well as anomaly detection
    methods. Ideally, the goal is to build a time series forecasting model that is robust to the
    anomalous events and generalise well on normal operating conditions.

    """

    def __init__(self):
        super().__init__(
            url="https://maxhalford.github.io/files/datasets/web-traffic.csv.zip",
            filename="web-traffic.csv",
            task=base.MO_REG,
            n_features=3,
            n_outputs=2,
            n_samples=44_160,
            size=2_769_905,
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            dialect=PipeCSVDialect,
            target=["sessionsA", "sessionsB"],
            converters={
                "region": str,
                "isMissing": lambda x: x == "1.0",
                "sessionsA": lambda x: float(x) if x and x != "0.0" else None,
                "sessionsB": lambda x: float(x) if x and x != "0.0" else None,
            },
            parse_dates={"dateTime": "%Y-%m-%d %H:%M:%S"},
        )
