import numpy as np
from skmultiflow.classification.core.driftdetection.adwin import ADWIN


def test_adwin():
    """ test_adwin

    This test will insert data from a stream with concept drift into ADWIN.
    The data stream is contains in the first half a sequence of randomly generated 0's and 1's.
    Then, from index 999 to 1999 the sequence is changed to a normal distribution of integers from 0 to 7.

    """
    adwin = ADWIN()
    data_stream = np.load('drift_stream.npy')
    adwin_indices_expected = [1023, 1055, 1087, 1151]
    adwin_indices = []

    for i in range(data_stream.size):
        adwin.add_element(data_stream[i])
        if adwin.detected_change():
            adwin_indices.append(i)

    assert adwin_indices == adwin_indices_expected
