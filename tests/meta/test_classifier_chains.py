
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta.classifier_chains import ClassifierChain, MCC, ProbabilisticClassifierChain
from skmultiflow.data import make_logical
from sklearn.linear_model import SGDClassifier

import numpy as np


def test_classifier_chains():

    stream = MultilabelGenerator(random_state=112, n_targets=3, n_samples=5150)
    stream.prepare_for_use()
    estimator = SGDClassifier(random_state=112, max_iter=10)
    learner = ClassifierChain(base_estimator=estimator, random_state=112)

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
    expected_predictions = [[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0]]

    expected_correct_predictions = 21

    assert np.alltrue(np.array_equal(predictions, expected_predictions))
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray
    #  assert type(learner.predict_proba(X)) == np.ndarray  Not available because default loss is set to 'hinge'


def test_classifier_chains_all():
    seed = 1
    X, Y = make_logical(random_state=seed)

    # CC
    cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=seed))
    cc.fit(X, Y)
    y_predicted = cc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)

    # RCC
    rcc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=seed), order='random', random_state=seed)
    rcc.fit(X, Y)
    rcc.fit(X, Y)
    y_predicted = rcc.predict(X)
    y_expected = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)

    # MCC
    mcc = MCC(SGDClassifier(max_iter=100, loss='log', random_state=seed), M=1000)
    mcc.fit(X, Y)
    mcc.fit(X, Y)
    y_predicted = mcc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)

    # PCC
    pcc = ProbabilisticClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=seed))
    pcc.fit(X, Y)
    pcc.fit(X, Y)
    y_predicted = pcc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)
