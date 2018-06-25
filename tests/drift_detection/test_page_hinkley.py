import os
import numpy as np
from skmultiflow.drift_detection import PageHinkley


def test_page_hinkley(test_path):
    """
    ADWIN drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.
    """
    ph = PageHinkley()
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [28, 57, 86, 115, 145, 174, 203, 232, 262, 292, 322, 352, 382, 411, 441, 471, 500, 530, 560, 589,
                        618, 648, 678, 708, 737, 767, 796, 826, 856, 885, 914, 943, 973, 1002, 1031, 1060, 1090, 1120,
                        1150, 1179, 1208, 1237, 1266, 1295, 1325, 1354, 1383, 1413, 1443, 1472, 1502, 1532, 1562, 1591,
                        1620, 1649, 1678, 1708, 1738, 1768, 1798, 1828, 1857, 1887, 1916, 1946, 1975]
    detected_indices = []

    for i in range(data_stream.size):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            detected_indices.append(i)

    assert detected_indices == expected_indices
