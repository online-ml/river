import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import MultiTargetRegressionHoeffdingTree


def test_hoeffding_tree():
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='mean')

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = \
        np.array([[0., 0., 0.],
                  [-14.46855757, -12.83566952, -7.39262486],
                  [37.66624157, 53.39153959, 69.99462597],
                  [10.16720844, 27.90949035, 10.66612076],
                  [15.24183569, 45.09487078, 28.00371775],
                  [-1.23856867, 41.08227011, 14.57608716],
                  [-0.73485404, 35.84590851, 11.86116888],
                  [16.28638992, 26.82830652, 12.49151208],
                  [18.00466059, 21.43992139, 18.13982322],
                  [23.31599303, 16.55916914, 10.46846807],
                  [20.5077, 20.83298116, 3.38325343],
                  [9.77883729, 11.59183073, -7.29048781],
                  [-9.83544652, 6.47372288, -19.65806682],
                  [-8.10518583, 6.9628173, -15.74443581],
                  [3.67329629, 12.81426462, -9.98382323],
                  [3.30335228, 11.73810323, -14.41210144],
                  [4.01029233, 13.43531087, -10.62833255],
                  [8.80934025, 23.13277514, -5.68457949],
                  [7.43425052, 23.34571574, -8.21176418],
                  [8.83756451, 22.65667941, -5.0832087],
                  [6.30090888, 22.62531105, -5.05855469],
                  [2.5721736, 16.49211826, -11.60664902],
                  [6.44334982, 20.32497491, -7.44521979],
                  [14.08620334, 22.33328981, -1.25449572],
                  [13.48401275, 17.15537495, -2.96056151],
                  [15.16301426, 16.22998206, -2.39885879],
                  [17.40651997, 18.92037037, 0.69013327],
                  [19.85056712, 18.49590222, 3.37726255],
                  [18.10308407, 16.28719326, 0.74505587],
                  [19.20466077, 18.36866242, 4.0238999],
                  [22.90282187, 21.02966837, 10.94869776],
                  [27.11874279, 22.66181171, 16.69783325],
                  [29.11065132, 23.54317671, 16.13140312],
                  [29.74398137, 25.57254636, 18.25265686],
                  [27.35187536, 26.82403046, 15.89461989],
                  [23.59136931, 25.24830415, 11.55217932],
                  [23.08848361, 24.19249925, 10.51122408],
                  [23.69630925, 23.41103434, 12.55808398],
                  [22.79832171, 24.1647397, 11.76014802],
                  [26.20269904, 25.89890937, 14.86171067],
                  [24.37630894, 25.12954411, 13.21721064],
                  [21.18365252, 22.28206033, 11.41276487],
                  [20.03027063, 19.97878582, 9.57099565],
                  [18.04244096, 20.514763, 8.21283073],
                  [18.21722597, 21.05404534, 9.57112071],
                  [20.49346593, 22.11613268, 11.34386402],
                  [19.5484046, 20.88892996, 10.37050252],
                  [20.64818042, 22.59394355, 12.41280043],
                  [19.23967786, 21.71486955, 11.01648366],
                  [18.18133786, 20.81077752, 8.34907456]])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 167.40626294018753
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: mean - nb_threshold: 0 - ' \
        'nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_hoeffding_tree_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='perceptron',
                                                 random_state=1)

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = \
        np.array(
            [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
             [-1.49640531e+02, -5.83578420e+01, -9.21304080e+01],
             [2.35485309e+01, 8.32762191e+01, 9.81812328e+01],
             [-1.49219190e+01, 2.06541228e+01, -3.28862262e+01],
             [-1.78729523e+01, 7.42771340e+01, -9.11965971e+01],
             [2.52860869e+01, 7.29832421e+01, 4.57735442e+01],
             [8.10809954e+01, 9.45615567e+01, 1.67783038e+01],
             [1.73667037e+02, 2.23254110e+01, 7.33801928e+01],
             [-3.38676393e+01, -2.22606061e+01, -9.55952000e+01],
             [-3.91879123e+01, 6.13314468e+01, -1.11493541e+01],
             [-6.81161700e+01, -4.09051599e+01, -5.28573548e+01],
             [-6.52559332e+01, -9.42291294e+00, -3.79517209e+01],
             [-2.70515326e+01, -3.10492002e+01, -6.30454389e+01],
             [-1.33306872e+02, -4.34940108e+01, -9.11707089e+01],
             [7.99863009e+01, -3.64626540e+00, -3.44584032e+01],
             [6.91161065e+01, 8.02105971e+01, 5.94654004e+01],
             [6.72386399e+01, 2.13342159e+01, 2.25836144e+01],
             [-5.77512265e+00, -2.41853376e+01, -6.14194210e+01],
             [-4.58271509e+01, 2.52874516e+01, -5.46249450e+01],
             [1.08061792e+02, 1.57907339e+02, 7.11751201e+01],
             [-8.16539971e+00, -3.75408097e+01, -2.03684277e+01],
             [5.59936127e+00, 8.47462814e+01, 9.61016628e+00],
             [4.09540700e+01, 1.23203805e+01, 4.39401633e+01],
             [-1.05775523e-01, -1.90512513e+01, -7.24670361e+01],
             [-1.08710011e+02, -7.25865707e+01, -9.18481415e+01],
             [2.52409815e+01, 2.91245532e+01, 1.71243894e+00],
             [2.36458251e+01, -1.50257619e+01, -3.20673327e+01],
             [3.97701490e+01, 2.02966669e+01, 2.79796985e+01],
             [1.03101977e+01, 3.05995526e+01, -2.01835199e+00],
             [-1.03093228e+02, -6.13966587e+01, -9.19075041e+01],
             [1.05451929e+02, 1.03610901e+02, 5.60157263e+01],
             [-3.04876588e+01, -4.53070124e+00, -6.17051609e+01],
             [5.81046353e+01, -1.38742372e+01, -2.46192106e+00],
             [2.91640464e+01, 4.66874643e+01, 4.49398082e+01],
             [-6.49156671e+01, 4.19602573e+01, -9.68607212e+01],
             [3.28570113e+01, 4.11492233e+00, 7.79758663e+01],
             [-3.59851153e+01, -6.03866520e+01, -5.13014428e+01],
             [3.08904155e+01, 4.14479676e+01, 1.06421876e+01],
             [1.30449068e+01, -5.57835732e+01, 5.99958280e+00],
             [-4.91476386e+01, -1.91296529e+01, -2.47830265e+01],
             [-3.46946785e+01, 6.53262554e+00, -8.18786051e+01],
             [-1.01339287e+01, -7.95133530e+01, 1.15413893e+01],
             [-1.64578219e+01, 2.72434481e+01, -6.08002150e+00],
             [-5.92252244e+01, -1.21564069e+01, -3.25826295e+01],
             [4.62809896e+00, -1.64727660e+01, 4.13685554e+01],
             [-1.18644511e+02, -8.53254109e+01, -1.26665467e+02],
             [9.20422591e+01, 5.44532647e+01, 5.83914975e+01],
             [1.42944699e+01, 5.67195089e+01, 2.03305316e+01],
             [5.38815950e+01, 8.17848855e+00, -3.07636579e+01],
             [1.06299358e+02, 7.68401208e+01, 4.69463359e+01]])
    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)
    expected_error = 134.5949775942447
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: perceptron - ' \
        'nb_threshold: 0 - nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_hoeffding_tree_adaptive():
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='adaptive',
                                                 random_state=1)

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = \
        np.array(
            [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
             [-1.44685576e+01, -1.28356695e+01, -7.39262486e+00],
             [2.35485309e+01, 8.32762191e+01, 6.99946260e+01],
             [-1.49219190e+01, 2.06541228e+01, -3.28862262e+01],
             [-1.78729523e+01, 7.42771340e+01, -9.11965971e+01],
             [2.52860869e+01, 7.29832421e+01, 4.57735442e+01],
             [8.10809954e+01, 9.45615567e+01, 1.67783038e+01],
             [1.73667037e+02, 2.23254110e+01, 7.33801928e+01],
             [-3.38676393e+01, -2.22606061e+01, -9.55952000e+01],
             [-3.91879123e+01, 6.13314468e+01, -1.11493541e+01],
             [-6.81161700e+01, -4.09051599e+01, -5.28573548e+01],
             [-6.52559332e+01, -9.42291294e+00, -3.79517209e+01],
             [-2.70515326e+01, -3.10492002e+01, -6.30454389e+01],
             [-1.33306872e+02, -4.34940108e+01, -9.11707089e+01],
             [7.99863009e+01, -3.64626540e+00, -3.44584032e+01],
             [6.91161065e+01, 8.02105971e+01, 5.94654004e+01],
             [6.72386399e+01, 2.13342159e+01, 2.25836144e+01],
             [-5.77512265e+00, -2.41853376e+01, -6.14194210e+01],
             [-4.58271509e+01, 2.52874516e+01, -5.46249450e+01],
             [1.08061792e+02, 1.57907339e+02, 7.11751201e+01],
             [-8.16539971e+00, -3.75408097e+01, -2.03684277e+01],
             [5.59936127e+00, 8.47462814e+01, 9.61016628e+00],
             [4.09540700e+01, 1.23203805e+01, 4.39401633e+01],
             [-1.05775523e-01, -1.90512513e+01, -7.24670361e+01],
             [-1.08710011e+02, -7.25865707e+01, -9.18481415e+01],
             [2.52409815e+01, 2.91245532e+01, 1.71243894e+00],
             [2.36458251e+01, -1.50257619e+01, -3.20673327e+01],
             [3.97701490e+01, 2.02966669e+01, 2.79796985e+01],
             [1.03101977e+01, 3.05995526e+01, -2.01835199e+00],
             [-1.03093228e+02, -6.13966587e+01, -9.19075041e+01],
             [1.05451929e+02, 1.03610901e+02, 5.60157263e+01],
             [-3.04876588e+01, -4.53070124e+00, -6.17051609e+01],
             [5.81046353e+01, -1.38742372e+01, -2.46192106e+00],
             [2.91640464e+01, 4.66874643e+01, 4.49398082e+01],
             [-6.49156671e+01, 4.19602573e+01, -9.68607212e+01],
             [3.28570113e+01, 4.11492233e+00, 7.79758663e+01],
             [-3.59851153e+01, -6.03866520e+01, -5.13014428e+01],
             [3.08904155e+01, 4.14479676e+01, 1.06421876e+01],
             [1.30449068e+01, -5.57835732e+01, 5.99958280e+00],
             [-4.91476386e+01, -1.91296529e+01, -2.47830265e+01],
             [-3.46946785e+01, 6.53262554e+00, -8.18786051e+01],
             [-1.01339287e+01, -7.95133530e+01, 1.15413893e+01],
             [-1.64578219e+01, 2.72434481e+01, -6.08002150e+00],
             [-5.92252244e+01, -1.21564069e+01, -3.25826295e+01],
             [4.62809896e+00, -1.64727660e+01, 4.13685554e+01],
             [-1.18644511e+02, -8.53254109e+01, -1.26665467e+02],
             [9.20422591e+01, 5.44532647e+01, 5.83914975e+01],
             [1.42944699e+01, 5.67195089e+01, 2.03305316e+01],
             [5.38815950e+01, 8.17848855e+00, -3.07636579e+01],
             [1.06299358e+02, 7.68401208e+01, 4.69463359e+01]])
    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)
    expected_error = 134.43981366919218
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: adaptive - ' \
        'nb_threshold: 0 - nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_hoeffding_tree_coverage(test_path):
    # Cover nominal attribute observer
    test_file = os.path.join(test_path, 'multi_target_regression_data.npz')
    data = np.load(test_file)
    X = data['X']
    Y = data['Y']

    learner = MultiTargetRegressionHoeffdingTree(
                leaf_prediction='mean',
                nominal_attributes=[i for i in range(3)]
              )
    learner.partial_fit(X, Y)
