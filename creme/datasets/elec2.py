from .. import stream

from . import base


class Elec2(base.Dataset):
    """Electricity prices in New South Wales.

    This data was collected from the Australian New South Wales Electricity Market. In this market,
    prices are not fixed and are affected by demand and supply of the market. They are set every
    five minutes. Electricity transfers to/from the neighboring state of Victoria were done to
    alleviate fluctuations.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `SPLICE-2 Comparative Evaluation: Electricity Pricing <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405>`_
        2. `DataHub description <https://datahub.io/machine-learning/electricity#readme>`_

    """

    def __init__(self, data_home=None, verbose=False):
        super().__init__(
            n_samples=45_312,
            n_features=8,
            category=base.BINARY_CLF,
            name='electricity',
            url='https://maxhalford.github.io/files/datasets/electricity.zip',
            data_home=data_home,
            archive_type='zip',
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/electricity.csv',
            target_name='class',
            converters={
                'date': float,
                'day': int,
                'period': float,
                'nswprice': float,
                'nswdemand': float,
                'vicprice': float,
                'vicdemand': float,
                'transfer': float,
                'class': lambda x: x == 'UP'
            }
        )
