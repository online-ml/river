from river import stream

from . import base


class TrumpApproval(base.FileDataset):
    """Donald Trump approval ratings.

    This dataset was obtained by reshaping the data used by FiveThirtyEight for analyzing Donald
    Trump's approval ratings. It contains 5 features, which are approval ratings collected by
    5 polling agencies. The target is the approval rating from FiveThirtyEight's model. The goal of
    this task is to see if we can reproduce FiveThirtyEight's model.

    References
    ----------
    [^1]: [Trump Approval Ratings](https://projects.fivethirtyeight.com/trump-approval-ratings/)

    """

    def __init__(self):
        super().__init__(
            n_samples=1001,
            n_features=6,
            task=base.REG,
            filename="trump_approval.csv.gz",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="five_thirty_eight",
            converters={
                "ordinal_date": int,
                "gallup": float,
                "ipsos": float,
                "morning_consult": float,
                "rasmussen": float,
                "you_gov": float,
                "five_thirty_eight": float,
            },
        )
