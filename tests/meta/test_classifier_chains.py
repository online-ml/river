
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta.classifier_chains import ClassifierChain
from sklearn.linear_model import SGDClassifier

import numpy as np


def test_classifier_chains():

    stream = MultilabelGenerator(random_state=112)
    stream.prepare_for_use()
    estimator = SGDClassifier(random_state=112)
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
    expected_predictions = [[0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 1., 0., 1.],
                           [1., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 1.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 1.],
                           [0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 1.],
                           [1., 0., 1., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 1.],
                           [0., 1., 1., 0., 0.],
                           [1., 0., 1., 1., 1.],
                           [0., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 1.],
                           [0., 0., 1., 0., 0.],
                           [0., 0., 1., 0., 0.],
                           [0., 1., 0., 0., 1.],
                           [0., 0., 1., 0., 1.],
                           [1., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [1., 0., 1., 0., 0.],
                           [1., 0., 1., 0., 1.],
                           [0., 1., 0., 0., 1.],
                           [0., 1., 0., 1., 1.],
                           [1., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [1., 0., 1., 1., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [1., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1.],
                           [0., 0., 1., 1., 0.],
                           [1., 1., 1., 0., 1.],
                           [0., 0., 0., 0., 1.],
                           [1., 1., 0., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0.]]

    expected_correct_predictions = 16

    assert np.alltrue(np.array_equal(predictions, expected_predictions))
    assert correct_predictions == expected_correct_predictions

