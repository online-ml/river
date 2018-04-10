import os
import filecmp
import difflib
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout


def test_evaluate_holdout_classifier(tmpdir, test_path):
    # Setup file stream
    stream = RandomTreeGenerator(tree_seed=23, instance_seed=12, n_classes=4, n_nominal_attributes=2,
                                 n_numerical_attributes=5, n_values_per_nominal=5, max_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(stream.get_attributes_header()))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    # Setup evaluator
    n_wait = 200
    max_samples = 1000
    metrics = ['kappa', 'kappa_t', 'performance']
    output_file = os.path.join(str(tmpdir), "holdout_summary.csv")
    evaluator = EvaluateHoldout(n_wait=n_wait,
                                max_samples=max_samples,
                                test_size=50,
                                metrics=metrics,
                                output_file=output_file)

    # Evaluate
    result = evaluator.eval(stream=stream, model=learner)
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTree)

    assert learner.get_model_measurements == result_learner.get_model_measurements

    expected_file = os.path.join(test_path, 'holdout_summary.csv')
    compare_files(output_file, expected_file)


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
