import generation as gen
import files
import utils
import os
import models
import numpy as np
from tqdm.autonotebook import tqdm
import argparse

#########################################################
# Global random forests parameters
#########################################################

# the number of trees in the forest
n_estimators = 1000

# the minimum number of samples required to be at a leaf node
min_samples_leaf = 1

# the number of features to consider when looking for the best split
max_features = 6

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', default=0.1, type=float, nargs='?', const=0.1)
parser.add_argument('--noise', type=str, nargs = '?', const='ARMA', default='ARMA')
parser.add_argument('--reg', default='Friedman', type=str, nargs='?', const='Friedman')
parser.add_argument('--nrep', default=100, type=int, nargs='?', const=100)
parser.add_argument('--n', default=300, type=int, nargs='?', const=300)
parser.add_argument('--train', default=200, type=int, nargs='?', const=200)
parser.add_argument('--ar', nargs="*", type=float, const=None)
parser.add_argument('--ma', nargs="*", type=float, const=None)
parser.add_argument('--process_variance', type=int, nargs='?')
parser.add_argument('--scale', type=float, nargs='?')
parser.add_argument('--cores', default=1, type=int, nargs='?', const=1)

args = parser.parse_args()
params = vars(args)

alpha = params['alpha']
noise = params['noise']
regression = params['reg']
n_rep = params['nrep']
if params['ar'] is not None:
    ar = np.r_[1, params['ar']]
else:
    ar = [1]
if params['ma'] is not None:
    ma = np.r_[1, params['ma']]
else:
    ma = [1]
if 'process_variance' in params and params['process_variance'] is not None:
        params_noise = {'ar': ar, 'ma': ma, 'process_variance': params['process_variance']}
elif 'scale' in params and params['scale'] is not None:
        params_noise = {'ar': ar, 'ma': ma, 'scale': params['scale']}
else:
    params_noise = {'ar': ar, 'ma': ma}
n = params['n']
train_size = params['train']
test_size = n-train_size
cores = params['cores']

params_basemodel = {'n_estimators':n_estimators, 'min_samples_leaf':min_samples_leaf, 'max_features':max_features,
                    'cores': cores}

name = files.get_name_data(n, regression=regression, noise=noise, params_noise=params_noise, seed=n_rep)
if os.path.isfile('data/'+name+'.pkl'):
    data = files.load_file('data', name, 'pkl')
else:
    X, Y = gen.generate_multiple_data(n, params_noise=params_noise, seed_max=n_rep)
    data = {'X': X, 'Y': Y}
    files.write_file('data', name, 'pkl', data)

tab_gamma = [0,
             0.000005,
             0.00005,
             0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
             0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
             0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

results, methods = models.run_multiple_gamma_ACP(data, alpha, tab_gamma,
                                                 'RF',
                                                 params_basemodel, n_rep, regression,
                                                 noise, {}, params_noise, train_size)

for method in methods:
        name_dir, name_method = files.get_name_results(method, n, regression=regression, noise=noise,
                                                       params_noise=params_noise)
        results_method = results[method]
        files.write_file('results/'+name_dir, name_method, 'pkl', results_method)
