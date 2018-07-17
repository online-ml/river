from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.data.file_stream import FileStream
import os
import numpy as np

def test_KNN(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/datasets/sea_big.csv')
    stream = FileStream(test_file, -1, 1)
    stream.prepare_for_use()

    learner = KNN(k=8, max_window_size=2000, leaf_size=40)
    cnt = 0
    max_samples = 5000
    predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                            1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    assert np.alltrue(predictions == expected_predictions)