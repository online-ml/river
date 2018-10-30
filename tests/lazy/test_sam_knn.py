import numpy as np
from array import array
import os
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.data.file_stream import FileStream


def test_sam_knn(package_path):

    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_big.csv')

    stream = FileStream(test_file)
    stream.prepare_for_use()

    hyperParams = {'maxSize': 1000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': 'maxACCApprox',
                   'useLTM': False}

    learner = SAMKNN(n_neighbors=hyperParams['nNeighbours'], max_window_size=hyperParams['maxSize'],
                     weighting=hyperParams['knnWeights'],
                     stm_size_option=hyperParams['STMSizeAdaption'], use_ltm=hyperParams['useLTM'])

    cnt = 0
    max_samples = 5000
    predictions = array('d')

    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('d', [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                                       0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                       1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                                       0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                                       0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    assert np.alltrue(predictions == expected_predictions)

    assert type(learner.predict(X)) == np.ndarray
    #  assert type(learner.predict_proba(X)) == np.ndarray  predict_proba not implemented.


def test_sam_knn_coverage(package_path):

    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_big.csv')

    stream = FileStream(test_file)
    stream.prepare_for_use()

    hyperParams = {'maxSize': 50,
                   'n_neighbors': 3,
                   'weighting': 'uniform',
                   'stm_size_option': 'maxACC',
                   'min_stm_size': 10,
                   'useLTM': True}

    learner = SAMKNN(n_neighbors=hyperParams['n_neighbors'],
                     max_window_size=hyperParams['maxSize'],
                     weighting=hyperParams['weighting'],
                     stm_size_option=hyperParams['stm_size_option'],
                     min_stm_size=hyperParams['min_stm_size'],
                     use_ltm=hyperParams['useLTM'])

    cnt = 0
    max_samples = 1000
    predictions = array('i')

    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                                       0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                                       1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
                                       1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                                       0, 1, 0, 1, 1, 0, 1, 1, 1])
    assert np.alltrue(predictions == expected_predictions)
