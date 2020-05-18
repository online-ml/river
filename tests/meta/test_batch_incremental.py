from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.batch_incremental import BatchIncrementalClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import set_config
import numpy as np
# Force sklearn to show only the parameters whose default value have been changed when
# printing an estimator (backwards compatibility with versions prior to sklearn==0.23)
set_config(print_changed_only=True)


def test_batch_incremental():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)

    estimator = DecisionTreeClassifier(random_state=112)
    learner = BatchIncrementalClassifier(base_estimator=estimator, n_estimators=10)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

    cnt = 0
    max_samples = 5000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1

    performance = correct_predictions / len(predictions)
    expected_predictions = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]

    expected_correct_predictions = 31
    expected_performance = 0.6326530612244898

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray

    expected_info = "BatchIncrementalClassifier(base_estimator=DecisionTreeClassifier("\
                    "random_state=112), n_estimators=10, window_size=100)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info
