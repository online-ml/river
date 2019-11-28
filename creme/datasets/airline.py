import os

from .. import stream

from . import base


class Airline(base.Dataset):
    """Monthly number of international airline passengers.

    The stream contains 144 items and only one single feature, which is the month. The goal is to
    predict the number of passengers each month by capturing the trend and the seasonality of the
    data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `International airline passengers: monthly totals in thousands. Jan 49 – Dec 60 <https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line>`_

    """

    def __init__(self):
        super().__init__(
            n_samples=144,
            n_features=1,
            category=base.REG
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            os.path.join(directory, 'airline-passengers.csv'),
            target_name='passengers',
            converters={'passengers': int},
            parse_dates={'month': '%Y-%m'}
        )
