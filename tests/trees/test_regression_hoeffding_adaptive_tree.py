import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from difflib import SequenceMatcher


def test_hoeffding_tree():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner = HoeffdingAdaptiveTreeRegressor(leaf_prediction='mean', random_state=1)

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

    expected_info = "HoeffdingAdaptiveTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='mean', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_tree_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner = HoeffdingAdaptiveTreeRegressor(leaf_prediction='perceptron', random_state=1)

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
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 152.12931270533377
    assert np.isclose(error, expected_error)

    expected_info = "HoeffdingAdaptiveTreeRegressor(binary_split=False, grace_period=200, " \
                    "leaf_prediction='perceptron', learning_ratio_const=True, learning_ratio_decay=0.001, " \
                    "learning_ratio_perceptron=0.02, max_byte_size=33554432, memory_estimate_period=1000000, " \
                    "nb_threshold=0, no_preprune=False, nominal_attributes=None, random_state=1, " \
                    "remove_poor_atts=False, split_confidence=1e-07, stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray

    assert learner._estimator_type == 'regressor'


def test_regression_hoeffding_adaptive_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last column (used only in the multi-target regression case)
    stream = stream[1000:, :-1]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = np.arange(8)
    # Typo in leaf prediction
    learner = HoeffdingAdaptiveTreeRegressor(
        nominal_attributes=nominal_attr_idx,
        leaf_prediction='percptron'
    )

    learner.partial_fit(X, y)

    expected_description = "if Attribute 1 = -1.0:\n" \
                           "  if Attribute 0 = -15.0:\n" \
                           "    Leaf = Statistics {0: 66.0000, 1: -164.9262, 2: 412.7679}\n" \
                           "  if Attribute 0 = 0.0:\n" \
                           "    Leaf = Statistics {0: 71.0000, 1: -70.3639, 2: 70.3179}\n" \
                           "  if Attribute 0 = 1.0:\n" \
                           "    Leaf = Statistics {0: 83.0000, 1: 0.9178, 2: 0.8395}\n" \
                           "  if Attribute 0 = 2.0:\n" \
                           "    Leaf = Statistics {0: 74.0000, 1: 73.6454, 2: 73.8353}\n" \
                           "  if Attribute 0 = 3.0:\n" \
                           "    Leaf = Statistics {0: 59.0000, 1: 75.2899, 2: 96.4856}\n" \
                           "  if Attribute 0 = -30.0:\n" \
                           "    Leaf = Statistics {0: 13.0000, 1: -40.6367, 2: 127.1607}\n" \
                           "if Attribute 1 = 0.0:\n" \
                           "  if Attribute 0 = -15.0:\n" \
                           "    Leaf = Statistics {0: 64.0000, 1: -158.0874, 2: 391.2359}\n" \
                           "  if Attribute 0 = 0.0:\n" \
                           "    Leaf = Statistics {0: 72.0000, 1: -0.4503, 2: 0.8424}\n" \
                           "  if Attribute 0 = 1.0:\n" \
                           "    Leaf = Statistics {0: 67.0000, 1: 68.0365, 2: 69.6664}\n" \
                           "  if Attribute 0 = 2.0:\n" \
                           "    Leaf = Statistics {0: 60.0000, 1: 77.7032, 2: 101.3210}\n" \
                           "  if Attribute 0 = 3.0:\n" \
                           "    Leaf = Statistics {0: 54.0000, 1: 77.4519, 2: 111.7702}\n" \
                           "  if Attribute 0 = -30.0:\n" \
                           "    Leaf = Statistics {0: 27.0000, 1: -83.8745, 2: 260.8891}\n" \
                           "if Attribute 1 = 1.0:\n" \
                           "  Leaf = Statistics {0: 412.0000, 1: 180.7178, 2: 1143.9712}\n" \
                           "if Attribute 1 = 2.0:\n" \
                           "  Leaf = Statistics {0: 384.0000, 1: 268.3498, 2: 1193.4180}\n" \
                           "if Attribute 1 = 3.0:\n" \
                           "  Leaf = Statistics {0: 418.0000, 1: 289.5005, 2: 1450.7667}\n"

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9


def test_hoeffding_adaptive_tree_regressor_alternate_tree():
    np.random.seed(8)

    learner = HoeffdingAdaptiveTreeRegressor(
        leaf_prediction='mean', grace_period=1000
    )

    max_samples = 7000
    cnt = 0

    p1 = False
    p2 = False

    while cnt < max_samples:
        X = [np.random.uniform(low=-1, high=1, size=2)]

        if cnt < 4000:
            if X[0][0] <= 0 and X[0][1] > 0:
                y = [np.random.normal(loc=-3, scale=1)]
            elif X[0][0] > 0 and X[0][1] > 0:
                y = [np.random.normal(loc=3, scale=1)]
            elif X[0][0] <= 0 and X[0][1] <= 0:
                y = [np.random.normal(loc=3, scale=1)]
            else:
                y = [np.random.normal(loc=-3, scale=1)]
        elif cnt < 5000:
            if not p1:
                expected_info = "if Attribute 1 <= -0.347867256929453:\n" \
                                "  Leaf = Statistics {0: 1310.0000, 1: 60.0637, 2: 13252.3632}\n" \
                                "if Attribute 1 > -0.347867256929453:\n" \
                                "  if Attribute 0 <= -0.010905749904186912:\n" \
                                "    Leaf = Statistics {0: 966.0000, 1: -1267.0737, 2: 9383.0230}\n" \
                                "  if Attribute 0 > -0.010905749904186912:\n" \
                                "    Leaf = Statistics {0: 1724.0000, 1: 1603.8751, 2: 17003.9844}\n" \

                assert expected_info == learner.get_model_description()
                p1 = True
            if X[0][0] <= 0 and X[0][1] > 0:
                y = [np.random.normal(loc=-10, scale=2)]
            elif X[0][0] > 0 and X[0][1] > 0:
                y = [np.random.normal(loc=10, scale=2)]
            elif X[0][0] <= 0 and X[0][1] <= 0:
                y = [np.random.normal(loc=3, scale=1)]
            else:
                y = [np.random.normal(loc=-3, scale=1)]
        else:
            if not p2:
                # Subtree turned into leaf (right branch from the root)
                expected_info = "if Attribute 1 <= -0.347867256929453:\n" \
                                "  if Attribute 0 <= -0.006772364899497507:\n" \
                                "    Leaf = Statistics {0: 683.0000, 1: 2035.8518, 2: 6826.7077}\n" \
                                "  if Attribute 0 > -0.006772364899497507:\n" \
                                "    Leaf = Statistics {0: 955.0000, 1: -1924.4740, 2: 9547.5316}\n" \
                                "if Attribute 1 > -0.347867256929453:\n" \
                                "  Leaf = Statistics {0: 350.0000, 1: 60.4979, 2: 27180.0527}\n"

                assert expected_info == learner.get_model_description()
                p2 = True
            if X[0][0] > 0:
                y = [np.random.normal(loc=20, scale=3)]
            else:
                y = [np.random.normal(loc=-20, scale=3)]

        learner.partial_fit(X, y)

        cnt += 1

    # Root node changed
    expected_info = "if Attribute 0 <= -0.0008180705425056001:\n" \
                    "  Leaf = Statistics {0: 851.0000, 1: -17061.1111, 2: 349640.3908}\n" \
                    "if Attribute 0 > -0.0008180705425056001:\n" \
                    "  Leaf = Statistics {0: 862.0000, 1: 17124.3350, 2: 349087.9636}\n" \

    assert expected_info == learner.get_model_description()
