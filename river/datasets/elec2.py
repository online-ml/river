from river import stream

from . import base


class Elec2(base.RemoteDataset):
    """Electricity prices in New South Wales.

    This is a binary classification task, where the goal is to predict if the price of electricity
    will go up or down.

    This data was collected from the Australian New South Wales Electricity Market. In this market,
    prices are not fixed and are affected by demand and supply of the market. They are set every
    five minutes. Electricity transfers to/from the neighboring state of Victoria were done to
    alleviate fluctuations.

    References
    ----------
    [^1]: [SPLICE-2 Comparative Evaluation: Electricity Pricing](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405)
    [^2]: [DataHub description](https://datahub.io/machine-learning/electricity#readme)

    """

    def __init__(self):
        super().__init__(
            url="https://maxhalford.github.io/files/datasets/electricity.zip",
            size=3091689,
            task=base.BINARY_CLF,
            n_samples=45_312,
            n_features=8,
            filename="electricity.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="class",
            converters={
                "date": float,
                "day": int,
                "period": float,
                "nswprice": float,
                "nswdemand": float,
                "vicprice": float,
                "vicdemand": float,
                "transfer": float,
                "class": lambda x: x == "UP",
            },
        )
