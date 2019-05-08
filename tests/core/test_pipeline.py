import os

from skmultiflow.lazy import KNNAdwin
from skmultiflow.core import Pipeline
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.transform import OneHotToCategorical

import numpy as np


def test_pipeline(test_path):
    n_categories = 5

    # Load test data generated using:
    # RandomTreeGenerator(tree_random_state=1, sample_random_state=1,
    #                     n_cat_features=n_categories, n_num_features=0)
    test_file = os.path.join(test_path, 'data-one-hot.npz')
    data = np.load(test_file)
    X = data['X']
    y = data['y']
    stream = DataStream(data=X, y=y)
    stream.prepare_for_use()

    # Setup transformer
    cat_att_idx = [[i + j for i in range(n_categories)] for j in range(0, n_categories * n_categories, n_categories)]
    transformer = OneHotToCategorical(categorical_list=cat_att_idx)

    # Set up the classifier
    classifier = KNNAdwin(n_neighbors=2, max_window_size=50, leaf_size=40)
    # Setup the pipeline
    pipe = Pipeline([('one-hot', transformer), ('KNNAdwin', classifier)])
    # Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=False, pretrain_size=10, max_samples=100)
    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)

    metrics = evaluator.get_mean_measurements()

    expected_accuracy = 0.5555555555555556
    assert np.isclose(expected_accuracy, metrics[0].get_accuracy())

    expected_kappa = 0.11111111111111116
    assert np.isclose(expected_kappa, metrics[0].get_kappa())
