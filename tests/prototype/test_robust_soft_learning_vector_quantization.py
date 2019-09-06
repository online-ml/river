import numpy as np
from array import array
from skmultiflow.data import SEAGenerator
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.core.base import is_classifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_rslvq():
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()

    learner_adadelta = RSLVQ(gradient_descent='adadelta')
    learner_vanilla = RSLVQ(gradient_descent='vanilla')

    cnt = 0
    max_samples = 5000
    y_pred_vanilla = array('i')
    y_pred_adadelta = array('i')
    X_batch = []
    y_batch = []
    wait_samples = 100

    # Check if predicted labels are as expected
    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred_vanilla.append(learner_vanilla.predict(X)[0])
            y_pred_adadelta.append(learner_adadelta.predict(X)[0])
        learner_adadelta.partial_fit(X, y, classes=stream.target_values)
        learner_vanilla.partial_fit(X, y, classes=stream.target_values)
        cnt += 1

    expected_predictions_vanilla = array('i',
                                         [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
                                          1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                                          0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                          0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
                                          0, 1])

    expected_predictions_adadelta = array('i',
                                          [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
                                           1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                                           0, 0, 1, 0, 1])

    assert np.alltrue(y_pred_vanilla == expected_predictions_vanilla)
    assert np.alltrue(y_pred_adadelta == expected_predictions_adadelta)

    # Check get_info method
    expected_info = "RobustSoftLearningVectorQuantization(gamma=0.9, gradient_descent='vanilla',\n" \
                    "                                     initial_prototypes=None,\n" \
                    "                                     prototypes_per_class=1, random_state=None,\n" \
                    "                                     sigma=1.0)"

    assert learner_vanilla.get_info() == expected_info

    # Check reset method
    learner_vanilla.reset()
    learner_vanilla.fit(X=np.array(X_batch[:4500]), y=np.array(y_batch[:4500]))

    learner_adadelta.reset()
    learner_adadelta.fit(X=np.array(X_batch[:4500]),
                         y=np.array(y_batch[:4500]))

    # Check classifiers performance
    learner_w_init_ppt = RSLVQ(initial_prototypes=[
                        [2.59922826, 2.57368134, 4.92501, 0],
                        [6.05801971, 6.01383352, 5.02135783, 1]],
                        gradient_descent='adadelta')
    learner_w_init_ppt.fit(X=np.array(X_batch[:4500]),
                           y=np.array(y_batch[:4500]))

    expected_score_ppt = .9539078156312625
    assert np.isclose(expected_score_ppt,
                      learner_w_init_ppt.score(X=np.array(X_batch[4501:]),
                                               y=np.array(y_batch[4501:])))

    expected_score_vanilla = .8897795591182365
    assert np.isclose(expected_score_vanilla,
                      learner_vanilla.score(X=np.array(X_batch[4501:]),
                                            y=np.array(y_batch[4501:])))

    expected_score_adadelta = .9458917835671342
    assert np.isclose(expected_score_adadelta,
                      learner_adadelta.score(X=np.array(X_batch[4501:]),
                                             y=np.array(y_batch[4501:])))

    # Check types
    assert is_classifier(learner_vanilla)
    assert is_classifier(learner_adadelta)

    assert type(learner_vanilla.predict(X)) == np.ndarray
    assert type(learner_adadelta.predict(X)) == np.ndarray

    # Check properties after learning
    expected_prototypes = np.array([[2.59922826, 2.57368134, 4.92501],
                                    [6.05801971, 6.01383352, 5.02135783]])

    assert np.allclose(learner_adadelta.prototypes, expected_prototypes)

    expected_prototypes_classes = np.array([0, 1])

    assert np.allclose(learner_adadelta.prototypes_classes, expected_prototypes_classes)

    expected_class_labels = np.array([0, 1])

    assert np.allclose(learner_adadelta.class_labels, expected_class_labels)
