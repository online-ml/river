from river import stream

from . import base


class ChickWeights(base.FileDataset):
    """Chick weights along time.

    The stream contains 578 items and 3 features. The goal is to predict the weight of each chick
    along time, according to the diet the chick is on. The data is ordered by time and then by
    chick.

    References
    ----------
    [^1]: [Chick weight dataset overview](http://rstudio-pubs-static.s3.amazonaws.com/107631_131ad1c022df4f90aa2d214a5c5609b2.html)

    """

    def __init__(self):
        super().__init__(
            filename="chick-weights.csv", n_samples=578, n_features=3, task=base.REG
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="weight",
            converters={"time": int, "weight": int, "chick": int, "diet": int},
        )
