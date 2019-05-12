from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.learn_pp import LearnPP
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def test_learn_pp():
    stream = RandomTreeGenerator(tree_random_state=2212, sample_random_state=2212)
    stream.prepare_for_use()
    estimator = DecisionTreeClassifier(random_state=2212)
    classifier = LearnPP(base_estimator=estimator, n_estimators=5, n_ensembles=5, random_state=2212)

    m = 200

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    # Pre training the classifier with 200 samples
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)
    predictions = []

    for i in range(10):
        X, y = stream.next_sample(200)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            corrects += np.sum(y == pred)
            predictions.append(pred[0])
        sample_count += m

    acc = corrects / sample_count

    expected_correct_predictions = 1138
    expected_acc = 0.569
    expected_predictions = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions
    assert type(classifier.predict(X)) == np.ndarray

    expected_info = "LearnPP(base_estimator=DecisionTreeClassifier(class_weight=None, " \
                    "criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, " \
                    "min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, " \
                    "min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, " \
                    "random_state=2212, splitter='best'), error_threshold=0.5, n_ensembles=5, " \
                    "n_estimators=5, random_state=2212, window_size=100)"
    info = " ".join([line.strip() for line in classifier.get_info().split()])
    assert info == expected_info
