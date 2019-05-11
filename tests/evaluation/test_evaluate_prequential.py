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
    assert np.isclose(mean_performance.get_accuracy(), expected_mean_accuracy)

    expected_mean_kappa = 0.231791
    assert np.isclose(mean_performance.get_kappa(), expected_mean_kappa)

    expected_mean_kappa_t = 0.236887
    assert np.isclose(mean_performance.get_kappa_t(), expected_mean_kappa_t)

    expected_current_accuracy = 0.430000
    assert np.isclose(current_performance.get_accuracy(), expected_current_accuracy)

    expected_current_kappa = 0.223909
    assert np.isclose(current_performance.get_kappa(), expected_current_kappa)

    expected_current_kappa_t = 0.240000
    assert np.isclose(current_performance.get_kappa_t(), expected_current_kappa_t)

    expected_info = "EvaluatePrequential(batch_size=1, data_points_for_classification=False,\n" \
                    "                    max_samples=1000, max_time=inf,\n" \
                    "                    metrics=['accuracy', 'kappa', 'kappa_t'], n_wait=200,\n" \
                    "                    output_file='prequential_summary.csv',\n" \
                    "                    pretrain_size=200, restart_stream=True, show_plot=False)"
    assert evaluator.get_info() == expected_info


def test_evaluate_classification_metrics():

    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    max_samples = 1000
    metrics = ['f1', 'precision', 'recall', 'gmean']
    evaluator = EvaluatePrequential(max_samples=max_samples,
                                    metrics=metrics)

    # Evaluate
    evaluator.evaluate(stream=stream, model=learner)
    mean_performance, current_performance = evaluator.get_measurements(model_idx=0)

    expected_current_f1_score = 0.7096774193548387
    expected_current_precision = 0.6814159292035398
    expected_current_recall = 0.7403846153846154
    expected_current_g_mean = 0.6802502367624613
    expected_mean_f1_score = 0.7009803921568628
    expected_mean_precision = 0.7185929648241206
    expected_mean_recall = 0.6842105263157895
    expected_mean_g_mean = 0.6954166367760247
    print(mean_performance.get_g_mean())
    print(mean_performance.get_recall())
    print(mean_performance.get_precision())
    print(mean_performance.get_f1_score())
    print(current_performance.get_g_mean())
    print(current_performance.get_recall())
    print(current_performance.get_precision())
    print(current_performance.get_f1_score())
    assert np.isclose(current_performance.get_f1_score(), expected_current_f1_score)
    assert np.isclose(current_performance.get_precision(), expected_current_precision)
    assert np.isclose(current_performance.get_recall(), expected_current_recall)
    assert np.isclose(current_performance.get_g_mean(), expected_current_g_mean)
    assert np.isclose(mean_performance.get_f1_score(), expected_mean_f1_score)
    assert np.isclose(mean_performance.get_precision(), expected_mean_precision)
    assert np.isclose(mean_performance.get_recall(), expected_mean_recall)
    assert np.isclose(mean_performance.get_g_mean(), expected_mean_g_mean)


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
