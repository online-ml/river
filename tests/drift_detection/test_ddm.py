import os
import numpy as np
from skmultiflow.drift_detection import DDM


def test_ddm(test_path):
    """
    DDM drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.
    """
    ddm = DDM()
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [1009]
    detected_indices = []

    for i in range(data_stream.size):
        ddm.add_element(data_stream[i])
        if ddm.detected_change():
            detected_indices.append(i)

    assert detected_indices == expected_indices
