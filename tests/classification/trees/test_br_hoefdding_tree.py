import numpy as np
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.trees.br_hoeffding_tree import BrHoeffdingTree
from skmultiflow.data.generators.multilabel_generator import MultilabelGenerator
import os

def test_br_hoeffding_tree(test_path):
    stream = MultilabelGenerator(n_samples=10000, n_features=15, n_targets=3, n_labels=4, random_state=112)

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
        if cnt % wait_samples == 0 and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        cnt += 1

    expected_predictions = [[0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1],
                            [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1],
                            [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1],
                            [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                            [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
                            [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1],
                            [1, 1, 1]]

    assert np.alltrue(predictions == expected_predictions)

    test_file = os.path.join(test_path, 'br_ht_prob.npy')
    data = np.load(test_file)

    assert np.allclose(proba_predictions, data)
