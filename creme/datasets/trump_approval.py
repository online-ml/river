import os

from .. import stream

from . import base


class TrumpApproval(base.FileDataset):
    """Donald Trump approval ratings.

    This dataset was obtained by reshaping the data used by FiveThirtyEight for analyzing Donald
    Trump's approval ratings. It contains 5 features, which are approval ratings collected by
    5 polling agencies. The target is the approval rating from FiveThirtyEight's model. The goal of
    this task is to see if we can reproduce FiveThirtyEight's model.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Trump Approval Ratings <https://projects.fivethirtyeight.com/trump-approval-ratings/>`_

    """

    def __init__(self):
        super().__init__(
            n_samples=1001,
            n_features=6,
            category=base.REG
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            os.path.join(directory, 'trump_approval.csv.gz'),
            target_name='five_thirty_eight',
            converters={
                'ordinal_date': int,
                'gallup': float,
                'ipsos': float,
                'morning_consult': float,
                'rasmussen': float,
                'you_gov': float,
                'five_thirty_eight': float
            }
        )
