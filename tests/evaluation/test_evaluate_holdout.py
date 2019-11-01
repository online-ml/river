import os
import filecmp
import difflib
import numpy as np
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluateHoldout


def test_evaluate_holdout_classifier(tmpdir, test_path):
    # Setup file stream
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    # Setup evaluator
    n_wait = 200
    max_samples = 1000
    metrics = ['accuracy', 'kappa', 'kappa_t']
    output_file = os.path.join(str(tmpdir), "holdout_summary.csv")
    evaluator = EvaluateHoldout(n_wait=n_wait,
                                max_samples=max_samples,
                                test_size=50,
                                metrics=metrics,
                                output_file=output_file)

    # Evaluate
    result = evaluator.evaluate(stream=stream, model=learner)
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTree)

    assert learner.get_model_measurements == result_learner.get_model_measurements

    expected_file = os.path.join(test_path, 'holdout_summary.csv')
    compare_files(output_file, expected_file)

    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_mean_accuracy = 0.344000
    assert np.isclose(mean_performance.accuracy_score(), expected_mean_accuracy)

    expected_mean_kappa = 0.135021
    assert np.isclose(mean_performance.kappa_score(), expected_mean_kappa)

    expected_mean_kappa_t = 0.180000
    assert np.isclose(mean_performance.kappa_t_score(), expected_mean_kappa_t)

    expected_current_accuracy = 0.360000
    assert np.isclose(current_performance.accuracy_score(), expected_current_accuracy)

    expected_current_kappa = 0.152542
    assert np.isclose(current_performance.kappa_score(), expected_current_kappa)

    expected_current_kappa_t = 0.200000
    assert np.isclose(current_performance.kappa_t_score(), expected_current_kappa_t)

    expected_info = "EvaluateHoldout(batch_size=1, dynamic_test_set=False, max_samples=1000,\n" \
                    "                max_time=inf, metrics=['accuracy', 'kappa', 'kappa_t'],\n" \
                    "                n_wait=200,\n" \
                    "                output_file='holdout_summary.csv',\n" \
                    "                restart_stream=True, show_plot=False, test_size=50)"
    assert evaluator.get_info() == expected_info


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
