import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from difflib import SequenceMatcher


def test_hoeffding_adaptive_tree_regressor_mean():
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

    expected_info = "HoeffdingAdaptiveTreeRegressor(binary_split=False, bootstrap_sampling=False, " \
                    "grace_period=200, leaf_prediction='mean', learning_ratio_const=True, " \
                    "learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, " \
                    "split_confidence=1e-07, stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_adaptive_tree_regressor_perceptron():
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

    expected_predictions = array('d', [207.20901655684412, 106.30316877540555, 101.46950096324191,
                                       114.38162776688861, 48.40271620592212, -79.94375846313639,
                                       -76.69182794940929, 88.38425569670662, -13.92372162581644,
                                       3.0549887923350507, 55.36276732455883, 32.0512081208464,
                                       17.54953203218902, -1.7305966738232161, 43.54548690756897,
                                       8.502241407478213, -61.14739038895263, 50.528736810827745,
                                       9.679668917948607, 89.93098085572623, 85.1994809437223,
                                       1.8721866382932664, -7.1972581323107825, -45.86230662663542,
                                       3.111671172363243, 57.921908276916646, 61.43400576850072,
                                       -16.61695641848216, -6.0769944259948065, 19.929266442289546,
                                       -60.972801351912224, -0.3342549973033524,
                                       -50.53334350658139, -14.885488543743078,
                                       -13.255920225124637, 28.909916365484275,
                                       -103.03499425386107, -36.44921969674884, -15.40018796932204,
                                       -84.98471039676006, 38.270205984888065, -62.97228157481581,
                                       -48.095864628804044, 95.5028130171316, 73.62390886812497,
                                       152.7135140597221, -120.4662342226783, -77.68182541723442,
                                       66.82059046110074])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)

    expected_error = 126.11208652969131
    assert np.isclose(error, expected_error)

    expected_info = "HoeffdingAdaptiveTreeRegressor(binary_split=False, " \
                    "bootstrap_sampling=False, grace_period=200, leaf_prediction='perceptron', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, " \
                    "learning_ratio_perceptron=0.02, max_byte_size=33554432, " \
                    "memory_estimate_period=1000000, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, " \
                    "split_confidence=1e-07, stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray

    assert learner._estimator_type == 'regressor'


def test_regression_hoeffding_adaptive_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last column (used only in the multi-target regression case)
    stream = stream[1500:, :-1]
    X, y = stream[:, :-1], stream[:, -1]
    X = X[:, :-1]

    nominal_attr_idx = np.arange(7)
    # Typo in leaf prediction
    learner = HoeffdingAdaptiveTreeRegressor(nominal_attributes=nominal_attr_idx,
                                             leaf_prediction='percptron',
                                             random_state=1)

    learner.partial_fit(X, y)

    expected_description = "if Attribute 0 = -15.0:\n" \
        "  if Attribute 1 = -1.0:\n" \
        "    Leaf = Statistics {0: 37.0000, 1: -92.4231, 2: 231.1636}\n" \
        "  if Attribute 1 = 0.0:\n" \
        "    Leaf = Statistics {0: 38.0000, 1: -94.0931, 2: 233.4825}\n" \
        "  if Attribute 1 = 1.0:\n" \
        "    Leaf = Statistics {0: 55.0000, 1: -131.1069, 2: 312.9920}\n" \
        "  if Attribute 1 = 2.0:\n" \
        "    Leaf = Statistics {0: 38.0000, 1: -90.3821, 2: 215.5215}\n" \
        "  if Attribute 1 = 3.0:\n" \
        "    Leaf = Statistics {0: 54.0000, 1: -124.1223, 2: 285.7867}\n" \
        "if Attribute 0 = 0.0:\n" \
        "  Leaf = Statistics {0: 123.0000, 1: 60.9178, 2: 132.5396}\n" \
        "if Attribute 0 = 1.0:\n" \
        "  Leaf = Statistics {0: 124.0000, 1: 134.7770, 2: 184.4009}\n" \
        "if Attribute 0 = 2.0:\n" \
        "  Leaf = Statistics {0: 104.0000, 1: 145.5842, 2: 212.8880}\n" \
        "if Attribute 0 = 3.0:\n" \
        "  Leaf = Statistics {0: 118.0000, 1: 186.4441, 2: 300.6575}\n" \
        "if Attribute 0 = -30.0:\n" \
        "  Leaf = Statistics {0: 88.0000, 1: -269.7967, 2: 828.2289}\n"

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9


def test_hoeffding_adaptive_tree_regressor_alternate_tree():
    learner = HoeffdingAdaptiveTreeRegressor(
        leaf_prediction='mean', grace_period=1000, random_state=7
    )

    np.random.seed(8)
    max_samples = 7000
    cnt = 0

    p1 = False
    p2 = False

    while cnt < max_samples:
        X = [np.random.uniform(low=-1, high=1, size=2)]

        if cnt < 3000:
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
                expected_info = "if Attribute 0 <= 0.7308480624289246:\n" \
                    "  if Attribute 1 <= 0.020068273107131107:\n" \
                    "    Leaf = Statistics {0: 900.0000, 1: 685.4441, 2: 9052.7232}\n" \
                    "  if Attribute 1 > 0.020068273107131107:\n" \
                    "    Leaf = Statistics {0: 1716.0000, 1: -284.7812, 2: 17014.5944}\n" \
                    "if Attribute 0 > 0.7308480624289246:\n" \
                    "  Leaf = Statistics {0: 384.0000, 1: -40.5676, 2: 3855.0453}\n"
                model_description = learner.get_model_description()
                assert expected_info == model_description
                p1 = True

            # Keep almost the same generation function
            if X[0][0] <= 0 and X[0][1] > 0:
                y = [np.random.normal(loc=-3, scale=1)]
            elif X[0][0] > 0 and X[0][1] > 0:
                y = [np.random.normal(loc=3, scale=1)]
            elif X[0][0] <= 0 and X[0][1] <= 0:
                y = [np.random.normal(loc=3, scale=1)]
            else:
                y = [np.random.normal(loc=-3, scale=1)]

            # But shift the normal mean in a specific region
            if X[0][0] <= 0.3:
                y = [np.random.normal(loc=5, scale=0.1)]
        elif cnt < 6000:
            if not p2:
                # Subtree swapped
                expected_info = "if Attribute 0 <= 0.7308480624289246:\n" \
                    "  if Attribute 0 <= 0.2979778083105622:\n" \
                    "    Leaf = Statistics {0: 1108.0000, 1: 5539.5153, 2: 27706.8525}\n" \
                    "  if Attribute 0 > 0.2979778083105622:\n" \
                    "    Leaf = Statistics {0: 342.0000, 1: 48.2119, 2: 3518.3529}\n" \
                    "if Attribute 0 > 0.7308480624289246:\n" \
                    "  Leaf = Statistics {0: 659.0000, 1: -28.8180, 2: 6546.5087}\n"

                assert expected_info == learner.get_model_description()
                p2 = True

            # Change how y is generated: only x_1 matters now
            if X[0][1] > 0:
                y = [np.random.normal(loc=20, scale=3)]
            else:
                y = [np.random.normal(loc=-20, scale=3)]

        learner.partial_fit(X, y)

        cnt += 1

    # Root node changed
    expected_info = "if Attribute 1 <= 0.02469103490619995:\n" \
        "  Leaf = Statistics {0: 941.0000, 1: -18769.1383, 2: 378390.2088}\n" \
        "if Attribute 1 > 0.02469103490619995:\n" \
        "  Leaf = Statistics {0: 900.0000, 1: -2030.2098, 2: 355715.9719}\n"

    assert expected_info == learner.get_model_description()
