import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeRegressor
from difflib import SequenceMatcher


def test_hoeffding_tree_regressor():
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

    expected_predictions = array('d', [525.7553636732247, 352.8160300365902, 224.80744320456478,
                                       193.72837054292074, 132.6059603765031, 117.06974933197759,
                                       114.53342429855932, 89.37195405567235, 57.85335051891305,
                                       60.00883955911155, 47.263185779784266, 25.17616431074491,
                                       17.43259526890146, 47.33468996498019, 22.83975208548138,
                                       -7.659282840823236, 8.564101665071064, 14.61585289361161,
                                       11.560941733770441, 13.70120291865976, 1.1938438210799651,
                                       19.01970713481836, 21.23459424444584, -5.667473522309328,
                                       -5.203149619381393, 28.726275200889173, 41.03406433337882,
                                       27.950322712127267, 21.267116786963925, 5.53344652490152,
                                       6.753264259267268, -2.3288137435962213, -10.492766334689875,
                                       -11.19641058176631, -20.134685945295644, -19.36581990084085,
                                       -38.26894947177957, -34.90246284430353, -11.019543212232008,
                                       -22.016714766708127, -18.710456277443544, -20.5568019328217,
                                       -2.636583876625667, 24.787714491718187, 29.325261678088406,
                                       45.31267371823666, -48.271054430207776, -59.7649172085901,
                                       48.22724814037523])
    # assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 152.12931270533377
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


def test_hoeffding_tree_regressor_coverage(test_path):
    # Cover nominal attribute observer
    test_file = os.path.join(test_path, 'regression_data.npz')
    data = np.load(test_file)
    X = data['X']
    y = data['y']

    # Typo in leaf prediction
    learner = HoeffdingTreeRegressor(
        leaf_prediction='percptron', nominal_attributes=[i for i in range(3)]
    )
    print(learner.split_criterion)
    # Invalid split_criterion
    learner.split_criterion = 'VR'
    learner.partial_fit(X, y)

    assert learner._estimator_type == 'regressor'


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
