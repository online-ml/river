import numpy as np

from river.drift.eddm import EDDM


def test_eddm():
    """
    EDDM drift detection test.
    The first half of the data contains a sequence corresponding to a normal distribution with mean 0 and sigma 0.1.
    The second half corresponds to a normal distribution with mean 0.5 and sigma 0.1.
    """
    eddm = EDDM()

    np.random.seed(1)
    mu, sigma = 0, 0.1  # mean and standard deviation
    d_1 = np.random.normal(mu, sigma, 1000) > 0
    mu, sigma = 0.5, 0.1  # mean and standard deviation
    d_2 = np.random.normal(mu, sigma, 1000) > 0
    data_stream = np.concatenate((d_1.astype(int), d_2.astype(int)))

    expected_indices = [64, 1135]
    detected_indices = []

    for i in range(data_stream.size):
        eddm.update(data_stream[i])
        if eddm.change_detected:
            detected_indices.append(i)

    assert detected_indices == expected_indices
