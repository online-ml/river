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
parser.add_argument('--methods', nargs="*", type=str)
parser.add_argument('--gamma', default=0.01, type=float, nargs='?', const=0.01)
parser.add_argument('--B', default=30, type=int, nargs='?', const=30)
parser.add_argument('--mean', default=1, type=int, nargs='?', const=1)
parser.add_argument('--online', default=1, type=int, nargs='?', const=1)
parser.add_argument('--randomized', default=0, type=int, nargs='?', const=0)
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
if 'process_variance' in params:
    params_noise = {'ar': ar, 'ma': ma, 'process_variance': params['process_variance']}
elif 'scale' in params:
    params_noise = {'ar': ar, 'ma': ma, 'scale': params['scale']}
else:
    params_noise = {'ar': ar, 'ma': ma}
methods = params['methods']
gamma = params['gamma']
B = params['B']
if params['mean'] == 1:
    mean = True
elif params['mean'] == 0:
    mean = False
if params['online'] == 1:
    online = True
elif params['online'] == 0:
    online = False
n = params['n']
train_size = params['train']
test_size = n-train_size
if params['randomized'] == 1:
    randomized = True
elif params['randomized'] == 0:
    randomized = False
cores = params['cores']
params_methods = {'gamma': gamma, 'B': B, 'mean':mean, 'online':online, 'randomized': randomized}

params_basemodel = {'n_estimators':n_estimators, 'min_samples_leaf':min_samples_leaf, 'max_features':max_features,
                    'cores': cores}

name = files.get_name_data(n, regression=regression, noise=noise, params_noise=params_noise, seed=n_rep)
if os.path.isfile('data/'+name+'.pkl'):
    data = files.load_file('data', name, 'pkl')
else:
    X, Y = gen.generate_multiple_data(n, params_noise=params_noise, seed_max=n_rep)
    data = {'X': X, 'Y': Y}
    files.write_file('data', name, 'pkl', data)

results, methods_ran = models.run_experiments(data, alpha, methods, params_methods, 'RF', params_basemodel,
                                              n_rep, regression, noise, {}, params_noise, train_size)

for method in methods_ran:
        name_dir, name_method = files.get_name_results(method, n, online=online, randomized=randomized, regression=regression, noise=noise,
                                                       params_noise=params_noise)
        results_method = results[method]
        files.write_file('results/'+name_dir, name_method, 'pkl', results_method)
