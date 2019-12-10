import numpy as np
from array import array
import os
from skmultiflow.data import ConceptDriftStream, SEAGenerator, HyperplaneGenerator
from skmultiflow.trees import HAT


def test_hat_mc(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)
    stream.prepare_for_use()

    learner = HAT(leaf_prediction='mc')

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

    expected_info = "HAT(binary_split=False, bootstrap_sampling=True, grace_period=200,\n" \
                    "    leaf_prediction='mc', max_byte_size=33554432,\n" \
                    "    memory_estimate_period=1000000, nb_threshold=0, no_preprune=False,\n" \
                    "    nominal_attributes=None, remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "    split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"
    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 398.0, 1.0: 1000.0}\n'

    assert (learner.get_model_description() == expected_model_1)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    stream.restart()
    X, y = stream.next_sample(5000)

    learner = HAT(max_byte_size=30, leaf_prediction='mc', grace_period=10)
    learner.partial_fit(X, y)


def test_hat_nb(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)
    stream.prepare_for_use()

    learner = HAT(leaf_prediction='nb')

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

    expected_info = "HAT(binary_split=False, bootstrap_sampling=True, grace_period=200,\n" \
                    "    leaf_prediction='nb', max_byte_size=33554432,\n" \
                    "    memory_estimate_period=1000000, nb_threshold=0, no_preprune=False,\n" \
                    "    nominal_attributes=None, remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "    split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"

    assert learner.get_info() == expected_info
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hat_nba(test_path):
    stream = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1, random_state=2)

    stream.prepare_for_use()

    learner = HAT(leaf_prediction='nba')

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

    expected_info = "HAT(binary_split=False, bootstrap_sampling=True, grace_period=200,\n" \
                    "    leaf_prediction='nba', max_byte_size=33554432,\n" \
                    "    memory_estimate_period=1000000, nb_threshold=0, no_preprune=False,\n" \
                    "    nominal_attributes=None, remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "    split_criterion='info_gain', stop_mem_management=False, tie_threshold=0.05)"

    assert learner.get_info() == expected_info
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hoeffding_adaptive_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last two columns (regression targets)
    stream = stream[:, :-2]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = np.arange(7).tolist()
    learner = HAT(nominal_attributes=nominal_attr_idx)

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
                           "  Leaf = Class 3.0 | {3.0: 65.0, 4.0: 55.0}\n"

    assert learner.get_model_description() == expected_description
