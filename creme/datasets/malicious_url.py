import glob
import os

from . import base


class MaliciousURL(base.FileDataset):
    """Malicious URLs dataset.

    This dataset contains features about URLs that are classified as malicious or not.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        .. [1] `Detecting Malicious URLs <http://www.sysnet.ucsd.edu/projects/url/>`_
        .. [2] `Identifying Suspicious URLs: An Application of Large-Scale Online Learning <http://cseweb.ucsd.edu/~jtma/papers/url-icml2009.pdf>`_

    """

    def __init__(self, data_home=None, verbose=True):
        super().__init__(
            n_samples=2_396_130,
            n_features=3_231_961,
            category=base.BINARY_CLF,
            url='http://www.sysnet.ucsd.edu/projects/url/url_svmlight.tar.gz',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):

        files = glob.glob(f'{directory}/url_svmlight/Day*.svm')
        files.sort(key=lambda x: int(os.path.basename(x).split('.')[0][3:]))

        def parse_libsvm_feature(f):
            k, v = f.split(':')
            return int(k), float(v)

        # There are 150 files, where each file corresponds to a day
        for file in files:
            with open(file) as f:
                for line in f:
                    # Each file has the libsvm format
                    elements = line.rstrip().split(' ')
                    y = elements.pop(0) == '+1'
                    x = dict(parse_libsvm_feature(f) for f in elements)
                    yield x, y
