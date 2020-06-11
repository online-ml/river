from skmultiflow.data import ConceptDriftStream
from skmultiflow.meta import StreamingRandomPatchesClassifier

import numpy as np

import pytest


def test_srp_randompatches():

    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='randompatches',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
                             1, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                             1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def test_srp_randomsubspaces():

    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='randomsubspaces',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                             0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
                             1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def test_srp_resampling():
    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='resampling',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
                             1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                             1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def test_srp_coverage():
    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='m',
                                               subspace_size=3,
                                               random_state=1)
    run_prequential_supervised(stream, learner, max_samples=100, n_wait=10)

    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='sqrtM1',
                                               random_state=1)
    run_prequential_supervised(stream, learner, max_samples=100, n_wait=10)

    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               subspace_mode='MsqrtM1',
                                               disable_drift_detection=True,
                                               disable_weighted_vote=True,
                                               random_state=1)
    run_prequential_supervised(stream, learner, max_samples=100, n_wait=10)

    learner = StreamingRandomPatchesClassifier(n_estimators=3,
                                               disable_background_learner=True,
                                               nominal_attributes=[3, 4, 5],
                                               random_state=1)
    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40)

    # Cover model reset and init ensemble calls from predict_proba and partial_fit
    learner.reset()
    assert learner.ensemble is None

    X, y = stream.next_sample()
    learner.partial_fit(X, y)
    learner.reset()
    X, y = stream.next_sample()
    assert learner.predict_proba(X)[0] == 0.0

    with pytest.raises(ValueError):
        _ = StreamingRandomPatchesClassifier(training_method='invalid')

    with pytest.raises(ValueError):
        _ = StreamingRandomPatchesClassifier(subspace_mode='invalid')


def run_prequential_supervised(stream, learner, max_samples, n_wait, y_expected=None):
    stream.restart()

    y_pred = np.zeros(max_samples // n_wait, dtype=np.int)
    y_true = np.zeros(max_samples // n_wait, dtype=np.int)
    j = 0

    for i in range(max_samples):
        X, y = stream.next_sample()
        # Test every n samples
        if i % n_wait == 0:
            y_pred[j] = int(learner.predict(X)[0])
            y_true[j] = int(y[0])
            j += 1
        learner.partial_fit(X, y)

    assert type(learner.predict(X)) == np.ndarray

    if y_expected is not None:
        assert np.alltrue(y_pred == y_expected)
