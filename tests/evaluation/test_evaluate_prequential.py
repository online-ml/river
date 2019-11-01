import os
import filecmp
import difflib
import numpy as np
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential


def test_evaluate_prequential_classifier(tmpdir, test_path):
    # Setup file stream
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    # Setup evaluator
    max_samples = 1000
    metrics = ['accuracy', 'kappa', 'kappa_t']
    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file)

    # Evaluate
    result = evaluator.evaluate(stream=stream, model=[learner])
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTree)

    assert learner.get_model_measurements == result_learner.get_model_measurements

    expected_file = os.path.join(test_path, 'prequential_summary.csv')
    compare_files(output_file, expected_file)

    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    # Simple test. Tests for metrics are placed in the corresponding test module.
    expected_mean_accuracy = 0.436250
    assert np.isclose(mean_performance.accuracy_score(), expected_mean_accuracy)

    expected_mean_kappa = 0.231791
    assert np.isclose(mean_performance.kappa_score(), expected_mean_kappa)

    expected_mean_kappa_t = 0.236887
    assert np.isclose(mean_performance.kappa_t_score(), expected_mean_kappa_t)

    expected_current_accuracy = 0.430000
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)

    expected_current_kappa = 0.223909
    assert np.isclose(current_performance.kappa_score(), expected_current_kappa)

    expected_current_kappa_t = 0.240000
    assert np.isclose(current_performance.kappa_t_score(), expected_current_kappa_t)

    expected_info = "EvaluatePrequential(batch_size=1, data_points_for_classification=False,\n" \
                    "                    max_samples=1000, max_time=inf,\n" \
                    "                    metrics=['accuracy', 'kappa', 'kappa_t'], n_wait=200,\n" \
                    "                    output_file='prequential_summary.csv',\n" \
                    "                    pretrain_size=200, restart_stream=True, show_plot=False)"
    assert evaluator.get_info() == expected_info


def test_evaluate_classification_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    # Learner
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    max_samples = 1000
    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    metrics = ['accuracy', 'kappa', 'kappa_t', 'kappa_m', 'f1', 'precision', 'recall', 'gmean', 'true_vs_predicted']
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_accuracy = 0.685
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)


def test_evaluate_regression_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    from skmultiflow.data import RegressionGenerator
    from skmultiflow.trees import RegressionHoeffdingTree

    max_samples = 1000

    # Stream
    stream = RegressionGenerator(n_samples=max_samples)
    stream.prepare_for_use()

    # Learner
    htr = RegressionHoeffdingTree()

    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    metrics = ['mean_square_error', 'mean_absolute_error']
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file)

    evaluator.evaluate(stream=stream, model=htr, model_names=['HTR'])


def test_evaluate_multi_target_classification_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    from skmultiflow.data import MultilabelGenerator
    from skmultiflow.meta import MultiOutputLearner

    max_samples = 1000

    # Stream
    stream = MultilabelGenerator(n_samples=max_samples, random_state=1)
    stream.prepare_for_use()

    # Learner
    mol = MultiOutputLearner()

    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    metrics = ['hamming_score', 'hamming_loss', 'exact_match', 'j_index']
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file)

    evaluator.evaluate(stream=stream, model=[mol], model_names=['MOL'])


def test_evaluate_multi_target_regression_coverage(tmpdir):
    from skmultiflow.data import RegressionGenerator
    from skmultiflow.trees import MultiTargetRegressionHoeffdingTree

    max_samples = 1000

    # Stream
    stream = RegressionGenerator(n_samples=max_samples, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=7)
    stream.prepare_for_use()

    # Learner
    mtrht = MultiTargetRegressionHoeffdingTree(leaf_prediction='adaptive')

    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    metrics = ['average_mean_square_error', 'average_mean_absolute_error', 'average_root_mean_square_error']
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file)

    evaluator.evaluate(stream=stream, model=mtrht, model_names=['MTRHT'])


def test_evaluate_coverage(tmpdir):
    from skmultiflow.data import SEAGenerator
    from skmultiflow.bayes import NaiveBayes

    max_samples = 1000

    # Stream
    stream = SEAGenerator(random_state =1)
    stream.prepare_for_use()

    # Learner
    nb = NaiveBayes()

    output_file = os.path.join(str(tmpdir), "prequential_summary.csv")
    metrics = ['running_time', 'model_size']
    evaluator = EvaluatePrequential(max_samples=max_samples,
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
