import numpy as np
from skmultiflow.classification.core.driftdetection.eddm import EDDM


def test_eddm():
    """
    EDDM drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.
    """
    eddm = EDDM()
    data_stream = np.load('drift_stream.npy')
    expected_indices = [51, 129, 291, 337, 396, 456, 523, 581, 675, 730, 851]
    detected_indices = []

    for i in range(1000):   # Only use the first half of stream (input is 0 or 1)
        eddm.add_element(data_stream[i])
        if eddm.detected_change():
            detected_indices.append(i)

    assert detected_indices == expected_indices
