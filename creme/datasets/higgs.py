from .. import stream

from . import base


class Higgs(base.FileDataset):
    """Higgs dataset.

    The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22)
    are kinematic properties measured by the particle detectors in the accelerator. The last seven
    features are functions of the first 21 features; these are high-level features derived by
    physicists to help discriminate between the two classes.

    Parameters:
        data_home: The directory where you wish to store the data.
        verbose: Whether to indicate download progress or not.

    References:
        1. [UCI page](https://archive.ics.uci.edu/ml/datasets/HIGGS)

    """

    def __init__(self, data_home: str = None, verbose=True):
        super().__init__(
            n_samples=11_000_000,
            n_features=28,
            category=base.BINARY_CLF,
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
            data_home=data_home,
            uncompress=False,
            verbose=verbose
        )

    def _stream_X_y(self, path):

        features = [
            'lepton pT', 'lepton eta', 'lepton phi',
            'missing energy magnitude', 'missing energy phi',
            'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag',
            'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag',
            'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
            'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
        ]

        return stream.iter_csv(
            path,
            fieldnames=['is_signal', *features],
            target='is_signal',
            converters={'is_signal': lambda x: x.startswith('1'), **{f: float for f in features}}
        )
