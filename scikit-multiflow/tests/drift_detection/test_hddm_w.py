import numpy as np
from river.drift.hddm_w import HDDM_W


def test_hddm_w():
    """
    HDDM_W drift detection test.
    The first half of the data contains a sequence corresponding to a normal distribution with mean 0 and sigma 0.1.
    The second half corresponds to a normal distribution with mean 0.5 and sigma 0.1.

    """
    hddm_w = HDDM_W()

    # Data
    np.random.seed(1)
    mu, sigma = 0, 0.1  # mean and standard deviation
    d_1 = np.random.normal(mu, sigma, 1000) > 0
    mu, sigma = 0.5, 0.1  # mean and standard deviation
    d_2 = np.random.normal(mu, sigma, 1000) > 0
    data_stream = np.concatenate((d_1.astype(int), d_2.astype(int)))

    expected_indices = [1018]
    detected_indices = []

    for i in range(data_stream.size):
        hddm_w.update(data_stream[i])
        if hddm_w.change_detected:
            detected_indices.append(i)

    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_w.reset()
    # Data
    mu, sigma = 0.0, 0.1  # mean and standard deviation
    d_1 = np.random.normal(mu, sigma, 500) > 0
    mu, sigma = 0.25, 0.1  # mean and standard deviation
    d_2 = np.random.normal(mu, sigma, 500) > 0
    mu, sigma = 0.0, 0.1  # mean and standard deviation
    d_3 = np.random.normal(mu, sigma, 500) > 0
    mu, sigma = 0.25, 0.1  # mean and standard deviation
    d_4 = np.random.normal(mu, sigma, 500) > 0
    data_stream = np.concatenate((d_1.astype(int), d_2.astype(int), d_3.astype(int), d_4.astype(int)))

    expected_indices = [518, 1513]
    detected_indices = []

    for i in range(data_stream.size):
        hddm_w.update(data_stream[i])
        if hddm_w.change_detected:
            detected_indices.append(i)

    assert detected_indices == expected_indices
