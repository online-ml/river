import numpy
import numpy as np
from array import array

import pandas as pd
from river import synth
from river.prototype import RobustSoftLearningVectorQuantization as RSLVQ

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_rslvq():
    stream = synth.SEA(seed=1)

    learner_adadelta = RSLVQ(gradient_descent='adadelta')
    learner_vanilla = RSLVQ(gradient_descent='vanilla')

    cnt = 0
    max_samples = 5000
    target_values = []
    if stream.task == 'Binary classification':
        target_values = [0, 1]
    y_pred_vanilla = array('i')
    y_pred_adadelta = array('i')
    X_batch = []
    y_batch = []
    wait_samples = 100
    # Check if predicted labels are as expected
    for X, y in stream.take(max_samples):
        X_batch.append(X)
        y_batch.append(y)

        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred_vanilla.append(learner_vanilla.predict_one(X))
            y_pred_adadelta.append(learner_adadelta.predict_one(X))

        learner_adadelta.learn_one(X, y, classes=target_values)
        learner_vanilla.learn_one(X, y, classes=target_values)
        cnt += 1

    expected_predictions_vanilla = array('i', [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
                                               0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
                                               0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])

    expected_predictions_adadelta = array('i', [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                                                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                                                0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    assert np.alltrue(y_pred_vanilla == expected_predictions_vanilla)
    assert np.alltrue(y_pred_adadelta == expected_predictions_adadelta)

    # Check classifiers performance
    learner_w_init_ppt = RSLVQ(initial_prototypes=[
        [2.59922826, 2.57368134, 4.92501, 0],
        [6.05801971, 6.01383352, 5.02135783, 1]],
        gradient_descent='adadelta')
    learner_w_init_ppt.learn_many(X=pd.DataFrame(X_batch[:4500]),
                                  y=pd.DataFrame(y_batch[:4500]))

    resultDiffs = np.subtract(y_batch[:4500], learner_w_init_ppt.predict_many(pd.DataFrame(X_batch[:4500])))
    actual_score = (resultDiffs == 0).sum() / 4500
    expected_score_ppt = 0.9528888888888889
    assert np.isclose(expected_score_ppt, actual_score)

    resultDiffs = np.subtract(y_batch[:4501], learner_vanilla.predict_many(pd.DataFrame(X_batch[:4501])))
    vanilla_score = (resultDiffs == 0).sum() / 4501
    expected_score_vanilla = 0.43257053988002664
    assert np.isclose(expected_score_vanilla, vanilla_score)

    resultDiffs = np.subtract(y_batch[:4501], learner_adadelta.predict_many(pd.DataFrame(X_batch[:4501])))
    adadelta_score = (resultDiffs == 0).sum() / 4501
    expected_score_adadelta = 0.6127527216174183
    assert np.isclose(expected_score_adadelta, adadelta_score)

    assert type(learner_vanilla.predict_one(X)) == numpy.int32
    assert type(learner_adadelta.predict_one(X)) == numpy.int32
    assert type(learner_vanilla.predict_many(pd.DataFrame(X_batch[:500]))) == numpy.ndarray
    assert type(learner_adadelta.predict_many(pd.DataFrame(X_batch[:500]))) == numpy.ndarray

    # Check properties after learning
    expected_prototypes = np.array([[1.73648149, 4.94257091, 4.78011796], [2.18377321, 7.9457862, 6.98779235]])
    assert np.allclose(learner_adadelta.prototypes, expected_prototypes)

    expected_prototypes_classes = np.array([0, 1])
    assert np.allclose(learner_adadelta.prototypes_classes, expected_prototypes_classes)

    expected_class_labels = np.array([0, 1])
    assert np.allclose(learner_adadelta.class_labels, expected_class_labels)

