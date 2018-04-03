import numpy as np
from skmultiflow.classification.core.driftdetection.ddm import DDM


def test_ddm():
    """
    EDDM drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.
    """
    ddm = DDM()
    data_stream = np.load('drift_stream.npy')
    expected_indices = [29]
    detected_indices = []

    for i in range(1000):   # Only use the first half of stream (input is 0 or 1)
        ddm.add_element(data_stream[i])
        if ddm.detected_change():
            detected_indices.append(i)

    assert detected_indices == expected_indices
