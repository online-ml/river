from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.batch_incremental import BatchIncremental

from sklearn.tree import DecisionTreeClassifier
import numpy as np


def test_batch_incremental():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)
    stream.prepare_for_use()
    estimator = DecisionTreeClassifier(random_state=112)
    learner = BatchIncremental(base_estimator=estimator, n_estimators=10)

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

    expected_info = "BatchIncremental(base_estimator=DecisionTreeClassifier(class_weight=None, " \
                    "criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, " \
                    "min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, " \
                    "min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=112, " \
                    "splitter='best'), n_estimators=10, window_size=100)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info
