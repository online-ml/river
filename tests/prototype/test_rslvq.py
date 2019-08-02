import numpy as np
from array import array
from skmultiflow.data import SEAGenerator
from skmultiflow.prototype.rslvq import RSLVQ
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

    expected_predictions_vanilla = array('i', [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 
                                               1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 
                                               0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 
                                               0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 
                                               1])

    expected_predictions_adadelta = array('i', [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                                1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
                                                1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 
                                                0, 0, 1, 0, 1])

    assert np.alltrue(y_pred_vanilla == expected_predictions_vanilla)
    assert np.alltrue(y_pred_adadelta == expected_predictions_adadelta)

    expected_info = "RSLVQ(gamma=None, gradient_descent='vanilla', initial_prototypes=None,\n      prototypes_per_class=1, random_state=None, sigma=1.0)"
    
    assert learner_vanilla.get_info() == expected_info

    learner_vanilla.reset()
    learner_vanilla.fit(X=np.array(X_batch[:4500]), y=np.array(y_batch[:4500]))
    
    learner_adadelta.reset()
    learner_adadelta.fit(X=np.array(X_batch[:4500]), y=np.array(y_batch[:4500]))

    expected_score_vanilla = .8897795591182365
    assert np.isclose(expected_score_vanilla, learner_vanilla.score(X=np.array(X_batch[4501:]),
    y=np.array(y_batch[4501:])))
            
    expected_score_adadelta = .9458917835671342
    assert np.isclose(expected_score_adadelta, learner_adadelta.score(X=np.array(X_batch[4501:]),
    y=np.array(y_batch[4501:])))
    
    assert is_classifier(learner_vanilla)
    assert is_classifier(learner_adadelta)

    assert type(learner_vanilla.predict(X)) == np.ndarray
    assert type(learner_adadelta.predict(X)) == np.ndarray
