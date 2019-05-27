import pytest

from sklearn.linear_model import SGDClassifier
from sklearn import __version__ as sklearn_version

from skmultiflow.data import MultilabelGenerator, make_logical
from skmultiflow.meta import ClassifierChain, MonteCarloClassifierChain, ProbabilisticClassifierChain

import numpy as np


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_classifier_chains():
    seed = 112
    stream = MultilabelGenerator(random_state=seed, n_targets=3, n_samples=5150)
    stream.prepare_for_use()
    estimator = SGDClassifier(random_state=seed, tol=1e-3, max_iter=10)
    learner = ClassifierChain(base_estimator=estimator, random_state=seed)
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

    if not sklearn_version.startswith("0.21"):
        expected_predictions = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        assert np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 21
        assert correct_predictions == expected_correct_predictions

        expected_info = "ClassifierChain(base_estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None,\n" \
                        "       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,\n" \
                        "       l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=10,\n" \
                        "       n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',\n" \
                        "       power_t=0.5, random_state=112, shuffle=True, tol=0.001,\n" \
                        "       validation_fraction=0.1, verbose=0, warm_start=False),\n" \
                        "                order=None, random_state=112)"
        assert learner.get_info() == expected_info

    else:
        expected_predictions = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0],
                                [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        assert np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 26
        assert correct_predictions == expected_correct_predictions

        expected_info = "ClassifierChain(base_estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None,\n" \
                        "              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,\n" \
                        "              l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=10,\n" \
                        "              n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,\n" \
                        "              random_state=112, shuffle=True, tol=0.001,\n" \
                        "              validation_fraction=0.1, verbose=0, warm_start=False),\n" \
                        "                order=None, random_state=112)"
        assert learner.get_info() == expected_info

    assert type(learner.predict(X)) == np.ndarray


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_classifier_chains_all():
    seed = 1
    X, Y = make_logical(random_state=seed)

    # CC
    cc = ClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed))
    cc.partial_fit(X, Y)
    y_predicted = cc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)
    assert type(cc.predict_proba(X)) == np.ndarray

    # RCC
    rcc = ClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed), order='random',
                          random_state=seed)
    rcc.partial_fit(X, Y)
    y_predicted = rcc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)

    # MCC
    mcc = MonteCarloClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed), M=1000)
    mcc.partial_fit(X, Y)
    y_predicted = mcc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)

    # PCC
    pcc = ProbabilisticClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed))
    pcc.partial_fit(X, Y)
    y_predicted = pcc.predict(X)
    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    assert np.alltrue(y_predicted == y_expected)
