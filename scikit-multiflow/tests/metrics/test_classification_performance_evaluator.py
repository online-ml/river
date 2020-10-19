import numpy as np

from river.metrics import _ClassificationReport
from river.metrics import _RollingClassificationReport
from river.metrics import _MLClassificationReport
from river.metrics import _RollingMLClassificationReport


def test_binary_classification():
    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    performance_evaluator = _ClassificationReport()
    [performance_evaluator.add_result(y_true[i], y_pred[i]) for i in range(len(y_true))]

    expected_accuracy_score = 90/100
    assert expected_accuracy_score == performance_evaluator.accuracy_score()

    expected_kappa_score = (expected_accuracy_score - 0.82) / (1 - 0.82)
    assert np.isclose(expected_kappa_score, performance_evaluator.kappa_score())

    expected_kappa_m_score = (expected_accuracy_score - .9) / (1 - 0.9)
    assert np.isclose(expected_kappa_m_score, performance_evaluator.kappa_m_score())

    expected_kappa_t_score = (expected_accuracy_score - .97) / (1 - 0.97)
    assert expected_kappa_t_score == performance_evaluator.kappa_t_score()

    expected_precision_score = 85 / (85+5)
    assert np.isclose(expected_precision_score, performance_evaluator.precision_score())

    expected_recall_score = 85 / (85+5)
    assert np.isclose(expected_recall_score, performance_evaluator.recall_score())

    expected_f1_score = 2 * ((expected_precision_score * expected_recall_score) /
                             (expected_precision_score + expected_recall_score))
    assert np.isclose(expected_f1_score, performance_evaluator.f1_score())

    expected_geometric_mean_score = np.sqrt((5 / (5 + 5)) * expected_recall_score)
    assert np.isclose(expected_geometric_mean_score, performance_evaluator.geometric_mean_score())

    expected_info = 'ClassificationPerformanceEvaluator(n_classes=2, n_samples=100, ' \
                    'accuracy_score=0.900000, kappa_m_score=0.000000, kappa_t_score=-2.333333, ' \
                    'kappa_m_score=0.000000, precision_score=0.944444, recall_score=0.944444, f1_score=0.944444, ' \
                    'geometric_mean_score=0.687184, majority_class=1)'
    assert expected_info == performance_evaluator.get_info()

    expected_last = (1, 0)
    assert expected_last == performance_evaluator.get_last()

    expected_majority_class = 1
    assert expected_majority_class == performance_evaluator.majority_class()

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0


def test_window_binary_classification():
    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    performance_evaluator = _RollingClassificationReport(window_size=20)
    for i in range(len(y_true)):
        performance_evaluator.add_result(y_true[i], y_pred[i])

    accuracy_score = 10/20
    assert accuracy_score == performance_evaluator.accuracy_score()

    expected_kappa_score = 0.0
    assert np.isclose(expected_kappa_score, performance_evaluator.kappa_score())

    expected_kappa_m_score = (.5 - (6/20))/(1 - (6/20))
    assert np.isclose(expected_kappa_m_score, performance_evaluator.kappa_m_score())

    expected_kappa_t_score = (.5 - (18/20))/(1 - (18/20))
    assert np.isclose(expected_kappa_t_score, performance_evaluator.kappa_t_score())

    expected_precision_score = 5 / (5 + 5)
    assert np.isclose(expected_precision_score, performance_evaluator.precision_score())

    expected_recall_score = 5 / (5 + 5)
    assert np.isclose(expected_recall_score, performance_evaluator.recall_score())

    expected_f1_score = 2 * ((expected_precision_score * expected_recall_score) /
                             (expected_precision_score + expected_recall_score))
    assert np.isclose(expected_f1_score, performance_evaluator.f1_score())

    expected_geometric_mean_score = np.sqrt((5 / (5 + 5)) * expected_recall_score)
    assert np.isclose(expected_geometric_mean_score, performance_evaluator.geometric_mean_score())

    expected_info = 'WindowClassificationPerformanceEvaluator(n_classes=2, window_size=20, n_samples=20, ' \
                    'accuracy_score=0.500000, kappa_m_score=0.285714, kappa_t_score=-4.000000, ' \
                    'kappa_m_score=0.285714, precision_score=0.500000, recall_score=0.500000, ' \
                    'f1_score=0.500000, geometric_mean_score=0.500000, majority_class=0)'
    assert expected_info == performance_evaluator.get_info()

    expected_last = (1.0, 0.0)
    assert expected_last == performance_evaluator.get_last()

    expected_majority_class = 0
    assert expected_majority_class == performance_evaluator.majority_class()

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0


def test_multi_class_classification():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2])

    performance_evaluator = _ClassificationReport()
    [performance_evaluator.add_result(y_true[i], y_pred[i]) for i in range(len(y_true))]

    expected_accuracy_score = 0.48
    assert expected_accuracy_score == performance_evaluator.accuracy_score()

    expected_kappa_score = 0.2545871559633027
    assert np.isclose(expected_kappa_score, performance_evaluator.kappa_score())

    expected_kappa_m_score = 0.13333333333333328
    assert np.isclose(expected_kappa_m_score, performance_evaluator.kappa_m_score())

    expected_kappa_t_score = -5.5000000000000036
    assert expected_kappa_t_score == performance_evaluator.kappa_t_score()

    expected_precision_score = 0.547008547008547
    assert np.isclose(expected_precision_score, performance_evaluator.precision_score())

    expected_recall_score = 0.5111111111111111
    assert np.isclose(expected_recall_score, performance_evaluator.recall_score())

    expected_f1_score = 0.46513720197930725
    assert np.isclose(expected_f1_score, performance_evaluator.f1_score())

    expected_geometric_mean_score = 0.446288633388113
    assert np.isclose(expected_geometric_mean_score, performance_evaluator.geometric_mean_score())

    expected_info = 'ClassificationPerformanceEvaluator(n_classes=3, n_samples=25, ' \
                    'accuracy_score=0.480000, kappa_m_score=0.133333, kappa_t_score=-5.500000, ' \
                    'kappa_m_score=0.133333, precision_score=0.547009, recall_score=0.511111, ' \
                    'f1_score=0.465137, geometric_mean_score=0.446289, majority_class=1)'

    assert expected_info == performance_evaluator.get_info()

    expected_last = (2, 2)
    assert expected_last == performance_evaluator.get_last()

    expected_majority_class = 1
    assert expected_majority_class == performance_evaluator.majority_class()

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0


def test_window_multi_class_classification():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2])

    performance_evaluator = _RollingClassificationReport(window_size=20)
    for i in range(len(y_true)):
        performance_evaluator.add_result(y_true[i], y_pred[i])

    accuracy_score = .4
    assert accuracy_score == performance_evaluator.accuracy_score()

    expected_kappa_score = 0.16955
    assert np.isclose(expected_kappa_score, performance_evaluator.kappa_score())

    expected_kappa_m_score = 0.2
    assert np.isclose(expected_kappa_m_score, performance_evaluator.kappa_m_score())

    expected_kappa_t_score = -5.0
    assert np.isclose(expected_kappa_t_score, performance_evaluator.kappa_t_score())

    expected_precision_score = 0.47222222222222215
    assert np.isclose(expected_precision_score, performance_evaluator.precision_score())

    expected_recall_score = 0.2888888888888889
    assert np.isclose(expected_recall_score, performance_evaluator.recall_score())

    expected_f1_score = 0.33785822021116135
    assert np.isclose(expected_f1_score, performance_evaluator.f1_score())

    expected_geometric_mean_score = 0.0
    assert np.isclose(expected_geometric_mean_score, performance_evaluator.geometric_mean_score())

    expected_info = 'WindowClassificationPerformanceEvaluator(n_classes=3, window_size=20, n_samples=20, ' \
                    'accuracy_score=0.400000, kappa_m_score=0.200000, kappa_t_score=-5.000000, ' \
                    'kappa_m_score=0.200000, precision_score=0.472222, recall_score=0.288889, ' \
                    'f1_score=0.337858, geometric_mean_score=0.000000, majority_class=1)'
    assert expected_info == performance_evaluator.get_info()

    expected_last = (2, 2)
    assert expected_last == performance_evaluator.get_last()

    expected_majority_class = 1
    assert expected_majority_class == performance_evaluator.majority_class()

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0


def test_multi_label_classification_measurements():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    performance_evaluator = _MLClassificationReport()
    for i in range(len(y_true)):
        performance_evaluator.add_result(y_true[i], y_pred[i])

    expected_exact_match_score = 0.85
    assert np.isclose(expected_exact_match_score, performance_evaluator.exact_match_score())

    expected_hamming_score = 1 - 0.06666666666666667
    assert np.isclose(expected_hamming_score, performance_evaluator.hamming_score())

    expected_hamming_loss_score = 0.06666666666666667
    assert np.isclose(expected_hamming_loss_score, performance_evaluator.hamming_loss_score())

    expected_jaccard_score = 0.9333333333333332
    assert np.isclose(expected_jaccard_score, performance_evaluator.jaccard_score())

    expected_info = 'MultiLabelClassificationPerformanceEvaluator(n_labels=3, n_samples=100, ' \
                    'hamming_score=0.933333, hamming_loss_score=0.066667, exact_match_score=0.850000, ' \
                    'jaccard_score=0.933333)'
    assert expected_info == performance_evaluator.get_info()

    expected_last_true = (1, 1, 1)
    expected_last_pred = (1, 0, 1)
    assert np.alltrue(expected_last_true == performance_evaluator.get_last()[0])
    assert np.alltrue(expected_last_pred == performance_evaluator.get_last()[1])

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0


def test_window_multi_label_classification_measurements():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    performance_evaluator = _RollingMLClassificationReport(window_size=20)
    for i in range(len(y_true)):
        performance_evaluator.add_result(y_true[i], y_pred[i])

    expected_exact_match_score = 0.25
    assert np.isclose(expected_exact_match_score, performance_evaluator.exact_match_score())

    expected_hamming_score = 1 - 0.33333333333333337
    assert np.isclose(expected_hamming_score, performance_evaluator.hamming_score())

    expected_hamming_loss_score = 0.33333333333333337
    assert np.isclose(expected_hamming_loss_score, performance_evaluator.hamming_loss_score())

    expected_jaccard_score = 0.6666666666666667
    assert np.isclose(expected_jaccard_score, performance_evaluator.jaccard_score())

    expected_info = 'WindowMultiLabelClassificationPerformanceEvaluator(n_labels=3, window_size=20, n_samples=20, ' \
                    'hamming_score=0.666667, hamming_loss_score=0.333333, exact_match_score=0.250000, ' \
                    'jaccard_score=0.666667)'
    assert expected_info == performance_evaluator.get_info()

    expected_last_true = (1, 1, 1)
    expected_last_pred = (1, 0, 1)
    assert np.alltrue(expected_last_true == performance_evaluator.get_last()[0])
    assert np.alltrue(expected_last_pred == performance_evaluator.get_last()[1])

    performance_evaluator.reset()
    assert performance_evaluator.n_samples == 0
