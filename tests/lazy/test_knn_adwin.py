from skmultiflow.lazy import KNNAdwin
from skmultiflow.data.file_stream import FileStream
import os
import numpy as np


def test_KNN_adwin(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_big.csv')
    stream = FileStream(test_file, -1, 1)
    stream.prepare_for_use()
    learner = KNNAdwin(n_neighbors=8, leaf_size=40, max_window_size=2000)

    cnt = 0
    max_samples = 5000
    predictions = []
    correct_predictions = 0

    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        learner.partial_fit(X, y)
        cnt += 1
    performance = correct_predictions / len(predictions)
    expected_predictions = [1, 0, 0, 1, 0,
                            0, 1, 1, 1, 0,
                            0, 1, 0, 0, 1,
                            0, 1, 0, 0, 1,
                            1, 0, 0, 1, 1,
                            1, 1, 1, 0, 1,
                            0, 1, 0, 1, 1,
                            0, 1, 0, 1, 0,
                            1, 1, 0, 1, 0,
                            1, 1, 1, 1]
    expected_correct_predictions = 40
    expected_performance = 0.8163265306122449

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions