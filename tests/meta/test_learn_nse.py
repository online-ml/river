import numpy as np

from sklearn.naive_bayes import GaussianNB

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import LearnNSE
from skmultiflow.trees import HoeffdingTree


def run_classifier(estimator, stream, pruning=None, ensemble_size=15, m=200):
    classifier = LearnNSE(base_estimator=estimator,
                          window_size=250,
                          pruning=pruning,
                          slope=0.5,
                          crossing_point=10,
                          n_estimators=ensemble_size)

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    # Pre training the classifier with 200 samples
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)

    for i in range(10):
        X, y = stream.next_sample(m)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            corrects += np.sum(y == pred)
        sample_count += m

    acc = corrects / sample_count

    assert type(classifier.predict(X)) == np.ndarray

    return corrects, acc, classifier


def test_learn_nse():
    stream = SEAGenerator(random_state=2212)
    stream.prepare_for_use()
    estimator = GaussianNB()

    corrects, acc, classifier = run_classifier(estimator, stream)

    expected_correct_predictions = 1754
    expected_acc = 0.877

    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions

    # Test reset method
    classifier.reset()
    assert len(classifier.ensemble) == 0
    assert len(classifier.ensemble_weights) == 0
    assert len(classifier.bkts) == 0
    assert len(classifier.wkts) == 0
    assert len(classifier.X_batch) == 0
    assert len(classifier.y_batch) == 0

    expected_info = 'LearnNSE(base_estimator=GaussianNB(priors=None, var_smoothing=1e-09),\n' \
                    '         crossing_point=10, n_estimators=15, pruning=None, slope=0.5,\n' \
                    '         window_size=250)'
    assert classifier.get_info() == expected_info
    # test pruning error
    corrects, acc, classifier = run_classifier(estimator, stream, pruning="error", ensemble_size=5)

    expected_correct_predictions = 1751
    expected_acc = 0.8755

    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions

    # test pruning age
    corrects, acc, classifier = run_classifier(estimator, stream, pruning="age", ensemble_size=5)

    expected_correct_predictions = 1774
    expected_acc = 0.887

    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions

    stream = SEAGenerator(random_state=2212)
    stream.prepare_for_use()

    estimator = HoeffdingTree()

    classifier = LearnNSE(base_estimator=estimator)

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    m = 250
    # Pre training the classifier
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)

    # print(classifier.ensemble_weights)
    for i in range(10):
        X, y = stream.next_sample(m)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            # print(pred)
            corrects += np.sum(y == pred)
        sample_count += m

    acc = corrects / sample_count
    expected_acc = 0.9436
    assert acc == expected_acc
