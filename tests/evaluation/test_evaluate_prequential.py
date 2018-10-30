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
    result = evaluator.evaluate(stream=stream, model=learner)
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTree)

    assert learner.get_model_measurements == result_learner.get_model_measurements

    expected_file = os.path.join(test_path, 'prequential_summary.csv')
    compare_files(output_file, expected_file)

    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)
    expected_mean_accuracy = 0.436250
    expected_mean_kappa = 0.231791
    expected_mean_kappa_t = 0.236887
    expected_current_accuracy = 0.430000
    expected_current_kappa = 0.223909
    expected_current_kappa_t = 0.240000
    assert np.isclose(mean_performance.get_accuracy(), expected_mean_accuracy)
    assert np.isclose(mean_performance.get_kappa(), expected_mean_kappa)
    assert np.isclose(mean_performance.get_kappa_t(), expected_mean_kappa_t)
    assert np.isclose(current_performance.get_accuracy(), expected_current_accuracy)
    assert np.isclose(current_performance.get_kappa(), expected_current_kappa)
    assert np.isclose(current_performance.get_kappa_t(), expected_current_kappa_t)


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
