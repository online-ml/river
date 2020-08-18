import os
import filecmp
import difflib
import random
import datetime

import numpy as np

import pytest

from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequentialDelayed
from skmultiflow.data import TemporalDataStream


def test_evaluate_prequential_delayed_classifier(tmpdir, test_path):
    # Setup file stream to generate data
    data = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4,
                               n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
                               max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15)
    # Number of samples to use
    max_samples = 1000

    # Get X and y
    X, y = data.next_sample(max_samples)
    y = y.astype(int)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=True)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(data.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    # Setup evaluator
    metrics = ['accuracy', 'kappa', 'kappa_t']
    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    # Evaluate
    result = evaluator.evaluate(stream=stream, model=[learner])
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTreeClassifier)

    assert learner.model_measurements == result_learner.model_measurements

    expected_file = os.path.join(test_path, 'prequential_delayed_summary.csv')
    compare_files(output_file, expected_file)

    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    # Simple test. Tests for metrics are placed in the corresponding test module.
    expected_mean_accuracy = 0.436250
    assert np.isclose(mean_performance.accuracy_score(), expected_mean_accuracy)

    expected_mean_kappa = 0.231791
    assert np.isclose(mean_performance.kappa_score(), expected_mean_kappa)

    expected_mean_kappa_t = 0.236886
    assert np.isclose(mean_performance.kappa_t_score(), expected_mean_kappa_t)

    expected_current_accuracy = 0.430000
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)

    expected_current_kappa = 0.223909
    assert np.isclose(current_performance.kappa_score(), expected_current_kappa)

    expected_current_kappa_t = 0.240000
    assert np.isclose(current_performance.kappa_t_score(), expected_current_kappa_t)

    expected_info = "EvaluatePrequentialDelayed(batch_size=1, " \
                    "data_points_for_classification=False, max_samples=1000, max_time=inf, " \
                    "metrics=['accuracy', 'kappa', 'kappa_t'], n_wait=200, " \
                    "output_file='prequential_delayed_summary.csv', pretrain_size=200, " \
                    "restart_stream=True, show_plot=False)"
    info = " ".join([line.strip() for line in evaluator.get_info().split()])
    assert info == expected_info


def test_evaluate_delayed_classification_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    data = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2,
                               n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
                               max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15)
    # Number of samples to use
    max_samples = 1000

    # Get X and y
    X, y = data.next_sample(max_samples)
    y = y.astype(int)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=False)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(data.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['accuracy', 'kappa', 'kappa_t', 'kappa_m',
               'f1', 'precision', 'recall', 'gmean', 'true_vs_predicted']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_accuracy = 0.705
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)


def test_evaluate_delayed_classification_single_sample_delay(tmpdir):
    # Test using a delay by samples
    data = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2,
                               n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
                               max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15)
    # Number of samples to use
    max_samples = 1000

    # Get X and y
    X, y = data.next_sample(max_samples)
    y = y.astype(int)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    with pytest.warns(UserWarning):
        stream = TemporalDataStream(X, y, time, sample_delay=50, ordered=False)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(data.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['accuracy', 'kappa', 'kappa_t', 'kappa_m',
               'f1', 'precision', 'recall', 'gmean', 'true_vs_predicted']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_accuracy = 0.7
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)


def test_evaluate_delayed_classification_single_time_delay(tmpdir):
    # Test using a single delay by time
    data = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2,
                               n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
                               max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15)
    # Number of samples to use
    max_samples = 1000

    # Get X and y
    X, y = data.next_sample(max_samples)
    y = y.astype(int)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, sample_delay=np.timedelta64(30, "D"), ordered=False)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(data.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['accuracy', 'kappa', 'kappa_t', 'kappa_m',
               'f1', 'precision', 'recall', 'gmean', 'true_vs_predicted']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_accuracy = 0.715
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)


def test_evaluate_delayed_classification_multiple_time_delay(tmpdir):
    # Test using multiple delays by time
    data = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2,
                               n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
                               max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15)
    # Number of samples to use
    max_samples = 1000

    # Get X and y
    X, y = data.next_sample(max_samples)
    y = y.astype(int)
    time = generate_random_dates(seed=1, samples=max_samples)
    delay = generate_random_delays(seed=1, samples=time)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, sample_delay=delay, ordered=False)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(data.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['accuracy', 'kappa', 'kappa_t', 'kappa_m',
               'f1', 'precision', 'recall', 'gmean', 'true_vs_predicted']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_accuracy = 0.715
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)


def test_evaluate_delayed_regression_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    from skmultiflow.data import RegressionGenerator
    from skmultiflow.trees import HoeffdingTreeRegressor

    max_samples = 1000

    # Generate data
    data = RegressionGenerator(n_samples=max_samples)
    # Get X and y
    X, y = data.next_sample(max_samples)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=False)

    # Learner
    htr = HoeffdingTreeRegressor()

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['mean_square_error', 'mean_absolute_error']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    evaluator.evaluate(stream=stream, model=htr, model_names=['HTR'])


def test_evaluate_delayed_multi_target_classification_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    from skmultiflow.data import MultilabelGenerator
    from skmultiflow.meta import MultiOutputLearner

    max_samples = 1000

    # Stream
    data = MultilabelGenerator(n_samples=max_samples, random_state=1)
    # Get X and y
    X, y = data.next_sample(max_samples)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=True)

    # Learner
    mol = MultiOutputLearner()

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['hamming_score', 'hamming_loss', 'exact_match', 'j_index']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    evaluator.evaluate(stream=stream, model=[mol], model_names=['MOL1'])


def test_evaluate_delayed_multi_target_regression_coverage(tmpdir):
    from skmultiflow.data import RegressionGenerator
    from skmultiflow.trees import iSOUPTreeRegressor

    max_samples = 1000

    # Stream
    data = RegressionGenerator(n_samples=max_samples, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=7)
    # Get X and y
    X, y = data.next_sample(max_samples)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=False)

    # Learner
    mtrht = iSOUPTreeRegressor(leaf_prediction='adaptive')

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['average_mean_square_error', 'average_mean_absolute_error',
               'average_root_mean_square_error']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           output_file=output_file)

    evaluator.evaluate(stream=stream, model=mtrht, model_names=['MTRHT'])


def test_evaluate_delayed_coverage(tmpdir):
    from skmultiflow.data import SEAGenerator
    from skmultiflow.bayes import NaiveBayes

    max_samples = 1000

    # Stream
    data = SEAGenerator(random_state=1)
    # Get X and y
    X, y = data.next_sample(max_samples)
    time = generate_random_dates(seed=1, samples=max_samples)

    # Setup temporal stream
    stream = TemporalDataStream(X, y, time, ordered=False)

    # Learner
    nb = NaiveBayes()

    output_file = os.path.join(str(tmpdir), "prequential_delayed_summary.csv")
    metrics = ['running_time', 'model_size']
    evaluator = EvaluatePrequentialDelayed(max_samples=max_samples,
                                           metrics=metrics,
                                           data_points_for_classification=True,
                                           output_file=output_file)

    evaluator.evaluate(stream=stream, model=nb, model_names=['NB'])


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True


def generate_random_dates(seed, samples):
    start = datetime.datetime(2020, 4, 30)
    end = datetime.datetime(2020, 7, 30)
    random.seed(seed)
    time = [random.random() * (end - start) + start for _ in range(samples)]
    return np.array(time, dtype="datetime64")


def generate_random_delays(seed, samples):
    random.seed(seed)
    delays = []
    for d in samples:
        delays.append(d + np.timedelta64(int(random.random() * 30), "D"))
    return np.array(delays, dtype="datetime64")
