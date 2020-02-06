from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.bayes import NaiveBayes
import numpy as np
from array import array


def test_awe():
    # prepare the stream
    stream = HyperplaneGenerator(random_state=1)

    # prepare the ensemble
    classifier = AccuracyWeightedEnsembleClassifier(n_estimators=5, n_kept_estimators=10,
                                                    base_estimator=NaiveBayes(),
                                                    window_size=200, n_splits=5)

    # test the classifier
    max_samples = 5000
    cnt = 0
    wait_samples = 100
    predictions = array('i')
    correct = 0
    while cnt < max_samples:
        X, y = stream.next_sample()
        pred = classifier.predict(X)
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(int(pred[0]))
        classifier.partial_fit(X, y)
        cnt += 1
        if pred[0] == y:
            correct += 1

    # assert model predictions
    expected_predictions = array('i', [
        0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 0, 0, 1, 1, 1, 1
    ])

    # assert model performance
    expected_accuracy = 0.875
    accuracy = correct / max_samples
    assert expected_accuracy == accuracy

    assert np.alltrue(predictions == expected_predictions)

    # assert model information
    expected_info = 'AccuracyWeightedEnsembleClassifier(base_estimator=NaiveBayes(nominal_attributes=None),\n' \
                    '                                   n_estimators=5, n_kept_estimators=10,\n' \
                    '                                   n_splits=5, window_size=200)'
    assert classifier.__repr__() == expected_info
