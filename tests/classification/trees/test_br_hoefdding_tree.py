import numpy as np
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.trees.br_hoeffding_tree import BrHoeffdingTree
from skmultiflow.data.generators.multilabel_generator import MultilabelGenerator
import os

def test_br_hoeffding_tree(test_path):
    stream = MultilabelGenerator(n_samples=10000, n_features=15, n_targets=3, n_labels=4, random_state=200)

    stream.prepare_for_use()

    learner = BrHoeffdingTree()

    cnt = 0
    max_samples = 5000
    predictions = []
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        learner.partial_fit(X, y)
        if cnt % wait_samples == 0:
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        cnt += 1

    expected_predictions = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1],
                           [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1],
                           [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0],
                           [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1],
                           [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0],
                           [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1],
                           [1, 1, 1], [1, 0, 1]]

    expected_proba_predictions_file_0 = os.path.join(test_path, 'br_pred_proba_0.npy')
    expected_proba_predictions_0 = np.load(expected_proba_predictions_file_0)
    #expected_proba_predictions_file_1 = os.path.join(test_path, 'pred_proba_1.npy')
    #expected_proba_predictions_1 = np.load(expected_proba_predictions_file_1)

    assert np.alltrue(proba_predictions == expected_proba_predictions_0)
    assert np.alltrue(predictions == expected_predictions)