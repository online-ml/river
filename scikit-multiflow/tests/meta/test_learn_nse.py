import pytest
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import set_config


from skmultiflow.data import SEAGenerator, RandomTreeGenerator
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.trees import HoeffdingTreeClassifier

# Force sklearn to show only the parameters whose default value have been changed when
# printing an estimator (backwards compatibility with versions prior to sklearn==0.23)
set_config(print_changed_only=True)


def run_classifier(estimator, stream, pruning=None, ensemble_size=15, m=200):
    classifier = LearnPPNSEClassifier(base_estimator=estimator,
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

    expected_info = 'LearnPPNSEClassifier(base_estimator=GaussianNB(), crossing_point=10, ' \
                    'n_estimators=15, pruning=None, slope=0.5, window_size=250)'
    info = " ".join([line.strip() for line in classifier.get_info().split()])
    assert info == expected_info
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

    estimator = HoeffdingTreeClassifier()

    classifier = LearnPPNSEClassifier(base_estimator=estimator)

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


def test_learn_nse_different_proba_sizes():
    m = 250
    stream = RandomTreeGenerator(
        tree_random_state=7, sample_random_state=8, n_classes=2
    )

    dt = DecisionTreeClassifier(random_state=7)
    classifier = LearnPPNSEClassifier(base_estimator=dt,
                                      window_size=250)

    # Pre training the classifier with 250 samples
    X, y = stream.next_sample(m)

    # Set manually classes
    classifier.partial_fit(X, y, classes=np.array([0, 1, 2, 3]))

    X, y = stream.next_sample(m)
    y[y == 0] = 3
    y[y == 1] = 2

    # pred = classifier.predict(X)
    classifier.partial_fit(X, y)

    X, y = stream.next_sample(m)
    y[y == 0] = 3

    pred = classifier.predict(X)
    classifier.partial_fit(X, y)

    if pred is not None:
        corrects = np.sum(y == pred)

    expected_correct_predictions = 115
    assert corrects == expected_correct_predictions

    stream.reset()
    # Repeating process with a skmultiflow-based learner
    ht = HoeffdingTreeClassifier(leaf_prediction='mc')
    classifier = LearnPPNSEClassifier(base_estimator=ht,
                                      window_size=250)

    # Pre training the classifier with 250 samples
    X, y = stream.next_sample(m)

    # Forcing exception to increase coverage
    with pytest.raises(RuntimeError):
        classifier.partial_fit(X, y, classes=None)

    classifier.reset()
    # Set manually classes
    classifier.partial_fit(X, y, classes=np.array([0, 1, 2, 3]))

    X, y = stream.next_sample(m)
    y[y == 0] = 3
    y[y == 1] = 2

    # pred = classifier.predict(X)
    classifier.partial_fit(X, y)

    X, y = stream.next_sample(m)
    y[y == 0] = 3

    pred = classifier.predict(X)

    if pred is not None:
        corrects = np.sum(y == pred)

    expected_correct_predictions = 109
    assert corrects == expected_correct_predictions
