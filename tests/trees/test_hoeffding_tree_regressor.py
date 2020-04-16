import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.utils import calculate_object_size
from difflib import SequenceMatcher


def test_hoeffding_tree_regressor_mean():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner = HoeffdingTreeRegressor(leaf_prediction='mean')

    cnt = 0
    max_samples = 500
    y_pred = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_true.append(y[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('d', [102.38946041769101, 55.6584574987656, 5.746076599168373, 17.11797209372667,
                                       2.566888222752787, 9.188247802192826, 17.87894804676911, 15.940629626883966,
                                       8.981172175448485, 13.152624115190092, 11.106058099429399, 6.473195313058236,
                                       4.723621479590173, 13.825568609556493, 8.698873073880696, 1.6452441811010252,
                                       5.123496188584294, 6.34387187194982, 5.9977733790395105, 6.874251577667707,
                                       4.605348088338317, 8.20112636572672, 9.032631648758098, 4.428189978974459,
                                       4.249801041367518, 9.983272668044492, 12.859518508979734, 11.741395774380285,
                                       11.230028410261868, 9.126921979081521, 9.132146661688296, 7.750655625124709,
                                       6.445145118245414, 5.760928671876355, 4.041291302080659, 3.591837600560529,
                                       0.7640424010500604, 0.1738639840537784, 2.2068337802212286, -81.05302946841077,
                                       96.17757415335177, -77.35894903819677, 95.85568683733698, 99.1981674250886,
                                       99.89327888035015, 101.66673013734784, -79.1904234513751, -80.42952143783687,
                                       100.63954789983896])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 143.11351404083086
    assert np.isclose(error, expected_error)

    expected_info = "HoeffdingTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='mean', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=None, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_tree_regressor_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner = HoeffdingTreeRegressor(leaf_prediction='perceptron', random_state=1)

    cnt = 0
    max_samples = 500
    y_pred = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_true.append(y[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('d', [-106.84237763060068, -10.965517384802226,
                                       -180.90711470797237, -218.20896751607663, -96.4271589961865,
                                       110.51551963099622, 108.34616947202511, 30.1720109214627,
                                       57.92205878998479, 77.82418885914053, 49.972060923364765,
                                       68.56117081695875, 15.996949915551697, -34.22744443808294,
                                       -19.762696110319702, -28.447329394752995,
                                       -50.62864370485592, -47.37357781048561, -99.82613515424342,
                                       13.985531117918336, 41.41709671929987, -34.679807275938174,
                                       62.75626094547859, 30.925078688018893, 12.130320819235365,
                                       119.3648998377624, 82.96422756064737, -6.920397563039609,
                                       -12.701774870569059, 24.883730398016034, -74.22855883237567,
                                       -0.8012436194087567, -83.03683748750394, 46.737839617687854,
                                       0.537404558240671, 48.53591837633138, -86.2259777783834,
                                       -24.985514024179967, 6.396035456152859, -90.19454995571908,
                                       32.05821807667601, -83.08553684151566, -28.32223999320023,
                                       113.28916673506842, 68.10498750807977, 173.9146410394573,
                                       -150.2067507947196, -74.10346402222962, 54.39153137687993])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 115.78916175164417
    assert np.isclose(error, expected_error)

    expected_info = "HoeffdingTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='perceptron', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_tree_regressor_coverage():
    max_samples = 1000
    max_size_mb = 2

    stream = RegressionGenerator(
        n_samples=max_samples, n_features=10, n_informative=7, n_targets=1,
        random_state=42
    )
    X, y = stream.next_sample(max_samples)

    # Cover memory management
    tree = HoeffdingTreeRegressor(
        leaf_prediction='mean', grace_period=100,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )
    tree.partial_fit(X, y)

    # A tree without memory management enabled reaches over 3 MB in size
    assert calculate_object_size(tree, 'MB') <= max_size_mb

    # Typo in leaf prediction
    tree = HoeffdingTreeRegressor(
        leaf_prediction='percptron', grace_period=100,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )
    # Invalid split_criterion
    tree.split_criterion = 'VR'

    tree.partial_fit(X, y)
    assert calculate_object_size(tree, 'MB') <= max_size_mb

    tree.reset()
    assert tree._estimator_type == 'regressor'


def test_hoeffding_tree_regressor_model_description():
    stream = RegressionGenerator(
        n_samples=500, n_features=20, n_informative=15, random_state=1
    )

    learner = HoeffdingTreeRegressor(leaf_prediction='mean')

    max_samples = 500
    X, y = stream.next_sample(max_samples)
    learner.partial_fit(X, y)

    expected_description = "if Attribute 6 <= 0.1394515530995348:\n" \
                           "  Leaf = Statistics {0: 276.0000, 1: -21537.4157, 2: 11399392.2187}\n" \
                           "if Attribute 6 > 0.1394515530995348:\n" \
                           "  Leaf = Statistics {0: 224.0000, 1: 22964.8868, 2: 10433581.2534}\n"

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9


def test_hoeffding_tree_regressor_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)

    # Remove class value
    stream = stream[:, np.delete(np.arange(8), 7)]
    # Removes the last column (used only in the multi-target regression case)
    stream = stream[:, :-1]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = np.arange(7).tolist()
    learner = HoeffdingTreeRegressor(nominal_attributes=nominal_attr_idx)

    learner.partial_fit(X, y)

    expected_description = "if Attribute 4 = 0.0:\n" \
                           "  Leaf = Statistics {0: 606.0000, 1: 1212.0000, 2: 3626.0000}\n" \
                           "if Attribute 4 = 1.0:\n" \
                           "  Leaf = Statistics {0: 551.0000, 1: 1128.0000, 2: 3400.0000}\n" \
                           "if Attribute 4 = 2.0:\n" \
                           "  Leaf = Statistics {0: 566.0000, 1: 1139.0000, 2: 3423.0000}\n" \
                           "if Attribute 4 = 3.0:\n" \
                           "  Leaf = Statistics {0: 577.0000, 1: 1138.0000, 2: 3374.0000}\n" \
                           "if Attribute 4 = 4.0:\n" \
                           "  Leaf = Statistics {0: 620.0000, 1: 1233.0000, 2: 3725.0000}\n" \
                           "if Attribute 4 = -3.0:\n" \
                           "  Leaf = Statistics {0: 80.0000, 1: 163.0000, 2: 483.0000}\n"

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9
