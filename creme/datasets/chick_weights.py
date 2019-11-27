import os

from .. import stream

from . import base


class ChickWeights(base.Dataset):
    """Chick weights along time.

    The stream contains 578 items and 3 features. The goal is to predict the weight of each chick
    along time, according to the diet the chick is on. The data is ordered by time and then by
    chick.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Chick Weight <http://rstudio-pubs-static.s3.amazonaws.com/107631_131ad1c022df4f90aa2d214a5c5609b2.html>`_

    """

    def __init__(self):
        super().__init__(
            n_samples=578,
            n_features=3,
            category=base.REG
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            os.path.join(directory, 'chick-weights.csv'),
            target_name='weight',
            converters={'time': int, 'weight': int, 'chick': int, 'diet': int}
        )
