import os

from .. import stream


def load_airline():
    """Returns a stream of monthly number of international airline passengers.

    The data is ordered by month.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    1. `International airline passengers: monthly totals in thousands. Jan 49 – Dec 60 <https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line>`_

    """
    return stream.iter_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'airline-passengers.csv'),
        target_name='passengers',
        types={'passengers': int},
        parse_dates={'month': '%Y-%m'}
    )
