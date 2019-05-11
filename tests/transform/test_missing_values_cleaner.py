import os
import numpy as np
from copy import copy
from skmultiflow.transform.missing_values_cleaner import MissingValuesCleaner


def test_missing_values_cleaner(test_path):

    test_file = os.path.join(test_path, 'data_nan.npy')
    X_nan = np.load(test_file)
    X = copy(X_nan)

    cleaner = MissingValuesCleaner(missing_value=np.nan, strategy='zero')

    X_complete = cleaner.transform(X)

    test_file = os.path.join(test_path, 'data_complete.npy')
    X_expected = np.load(test_file)
    assert np.alltrue(X_complete == X_expected)

    expected_info = "MissingValuesCleaner(missing_value=[nan], new_value=1, strategy='zero',\n" \
                    "                     window_size=200)"
    assert cleaner.get_info() == expected_info

    assert cleaner._estimator_type == 'transform'


def test_missing_values_cleaner_coverage(test_path):
    test_file = os.path.join(test_path, 'data_nan.npy')
    X_nan = np.load(test_file)
    X = copy(X_nan)

    cleaner = MissingValuesCleaner(missing_value=np.nan, strategy='mean', window_size=10)
    X = copy(X_nan)
    X_complete = cleaner.transform(X)

    cleaner = MissingValuesCleaner(missing_value=np.nan, strategy='mode', window_size=10)
    X = copy(X_nan)
    X_complete = cleaner.transform(X)

    cleaner = MissingValuesCleaner(missing_value=np.nan, strategy='median', window_size=10)
    X = copy(X_nan)
    X_complete = cleaner.transform(X)

    cleaner = MissingValuesCleaner(missing_value=np.nan, strategy='custom', new_value=-1)
    X = copy(X_nan)
    X_complete = cleaner.transform(X)

    cleaner = MissingValuesCleaner(missing_value=[np.nan], strategy='mean', new_value=-1)
    X = copy(X_nan)
    X_complete = cleaner.partial_fit(X)

    cleaner.partial_fit_transform(X=X)

