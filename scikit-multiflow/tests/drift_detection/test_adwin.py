import os
import numpy as np
from river.drift.adwin import ADWIN


def test_adwin(test_path):
    """
    ADWIN drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.

    """
    adwin = ADWIN()
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [1023, 1055, 1087, 1151]
    detected_indices = []

    for i in range(data_stream.size):
        adwin.update(data_stream[i])
        if adwin.change_detected:
            detected_indices.append(i)

    assert detected_indices == expected_indices

