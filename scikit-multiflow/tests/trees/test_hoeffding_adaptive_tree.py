import numpy as np
from array import array
import os
from skmultiflow.data import ConceptDriftStream, SEAGenerator, HyperplaneGenerator, \
    AGRAWALGenerator
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier


def test_hoeffding_adaptive_tree_mc(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)

    learner = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc', random_state=1)

    cnt = 0
    max_samples = 1000
    y_pred = array('i')
    y_proba = []
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_mc.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HoeffdingAdaptiveTreeClassifier(binary_split=False, bootstrap_sampling=True, grace_period=200, " \
                    "leaf_prediction='mc', max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, " \
                    "no_preprune=False, nominal_attributes=None, random_state=1, remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])

    assert info == expected_info

    expected_model_1 = 'Leaf = Class 1 | {0: 398.0, 1: 1000.0}\n'

    assert (learner.get_model_description() == expected_model_1)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    stream.restart()
    X, y = stream.next_sample(5000)

    learner = HoeffdingAdaptiveTreeClassifier(max_byte_size=30, leaf_prediction='mc', grace_period=10)
    learner.partial_fit(X, y)


def test_hoeffding_adaptive_tree_nb(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)

    learner = HoeffdingAdaptiveTreeClassifier(leaf_prediction='nb', random_state=1)

    cnt = 0
    max_samples = 1000
    y_pred = array('i')
    y_proba = []
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                                       0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                                       1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                                       0, 1, 1, 1, 1, 1, 0, 1, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_nb.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HoeffdingAdaptiveTreeClassifier(binary_split=False, bootstrap_sampling=True, grace_period=200, " \
                    "leaf_prediction='nb', max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, " \
                    "no_preprune=False, nominal_attributes=None, random_state=1, remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hoeffding_adaptive_tree_nba(test_path):
    stream = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1, random_state=2)

    learner = HoeffdingAdaptiveTreeClassifier(leaf_prediction='nba', random_state=1)

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    y_proba = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
                                       1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_nba.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HoeffdingAdaptiveTreeClassifier(binary_split=False, bootstrap_sampling=True, grace_period=200, " \
                    "leaf_prediction='nba', max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, " \
                    "no_preprune=False, nominal_attributes=None, random_state=1, remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hoeffding_adaptive_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last two columns (regression targets)
    stream = stream[:, :-2]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = np.arange(7).tolist()
    learner = HoeffdingAdaptiveTreeClassifier(nominal_attributes=nominal_attr_idx, random_state=1)

    learner.partial_fit(X, y, classes=np.unique(y))

    expected_description = "if Attribute 0 = -15.0:\n" \
                           "  Leaf = Class 2 | {2: 475.0}\n" \
                           "if Attribute 0 = 0.0:\n" \
                           "  Leaf = Class 0 | {0: 560.0, 1: 345.0}\n" \
                           "if Attribute 0 = 1.0:\n" \
                           "  Leaf = Class 1 | {0: 416.0, 1: 464.0}\n" \
                           "if Attribute 0 = 2.0:\n" \
                           "  Leaf = Class 1 | {0: 335.0, 1: 504.0}\n" \
                           "if Attribute 0 = 3.0:\n" \
                           "  Leaf = Class 1 | {0: 244.0, 1: 644.0}\n" \
                           "if Attribute 0 = -30.0:\n" \
                           "  Leaf = Class 3 | {3: 65.0, 4: 55.0}\n"

    assert learner.get_model_description() == expected_description


def test_hoeffding_adaptive_tree_alternate_tree():
    stream = AGRAWALGenerator(random_state=7)

    learner = HoeffdingAdaptiveTreeClassifier(random_state=1)

    cnt = 0
    change_point1 = 1500
    change_point2 = 2500
    change_point3 = 4000
    max_samples = 5000

    while cnt < max_samples:
        X, y = stream.next_sample()
        learner.partial_fit(X, y)
        cnt += 1

        if cnt > change_point1:
            stream.generate_drift()
            change_point1 = float('Inf')

            expected_description = "if Attribute 2 <= 63.63636363636363:\n" \
                                   "  if Attribute 2 <= 39.54545454545455:\n" \
                                   "    Leaf = Class 0 | {0: 397.5023676194098}\n" \
                                   "  if Attribute 2 > 39.54545454545455:\n" \
                                   "    if Attribute 2 <= 58.81818181818181:\n" \
                                   "      Leaf = Class 1 | {1: 299.8923824199619}\n" \
                                   "    if Attribute 2 > 58.81818181818181:\n" \
                                   "      Leaf = Class 0 | {0: 54.0, 1: 20.107617580038095}\n" \
                                   "if Attribute 2 > 63.63636363636363:\n" \
                                   "  Leaf = Class 0 | {0: 512.5755895049351}\n"
            assert expected_description == learner.get_model_description()

        if cnt > change_point2:
            stream.generate_drift()
            change_point2 = float('Inf')
            expected_description = "if Attribute 8 <= 268547.7178694747:\n" \
                                   "  Leaf = Class 0 | {0: 446.18690518790413, 1: 80.6180778406834}\n" \
                                   "if Attribute 8 > 268547.7178694747:\n" \
                                   "  Leaf = Class 1 | {0: 36.8130948120959, 1: 356.38192215931656}\n"
            assert expected_description == learner.get_model_description()

        if cnt > change_point3:
            stream.generate_drift()
            change_point3 = float('Inf')

    expected_description = "Leaf = Class 0 | {0: 1083.0, 1: 2.0}\n"
    assert expected_description == learner.get_model_description()
