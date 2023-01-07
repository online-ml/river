import river
import pickle
from utils import get_name_data, get_name_results

alpha = 0.1

tab_gamma=[0,0.000005,0.00005,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

# Simulations

n_rep = 500
n = 300
train_size = 200

import numpy as np

def aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg):
    test_size = n - train_size

    name = get_name_data(n, regression='Friedman', noise='ARMA', params_noise=params_noise, seed=n_rep)
    path = "/Users/mzaffran/Documents/Code/CP/cp-epf/data_cluster"
    with open(f"{path}/{name}.pkl", "rb") as f:
        data = pickle.load(f)

    methods = []
    for gamma in tab_gamma:
        methods.append(f'ACP_{gamma}')

    experts_low = np.empty((len(tab_gamma), n_rep, test_size))
    experts_high = np.empty((len(tab_gamma), n_rep, test_size))

    for idm in range(len(methods)):
        method = methods[idm]
        path = "/Users/mzaffran/Documents/Code/CP/cp-epf/results_cluster"
        names = get_name_results(method, n, regression='Friedman', noise='ARMA', params_noise=params_noise)
        with open(f"{path}/{names['directory']}/{names['method']}.pkl", "rb") as f:
            results = pickle.load(f)
        experts_low[idm,:,:] = results['Y_inf']
        experts_high[idm,:,:] = results['Y_sup']
        experts_low[idm,:,:][experts_low[idm,:,:] == -float("inf")] = -1000
        experts_high[idm,:,:][experts_high[idm,:,:] == float("inf")] = 1000

    experts_low_pred = np.empty((n_rep, test_size))
    experts_high_pred = np.empty((n_rep, test_size))

    for k in range(n_rep):
        mlpol_grad_low = mixture(Y=data['Y'][k,(train_size+1):data['Y'].shape[1]], experts=np.transpose(experts_low[:,k,:]), model=agg, loss_gradient=True, loss_type={'name':'pinball', 'tau':alpha/2})
        mlpol_grad_high = mixture(Y=data['Y'][k,(train_size+1):data['Y'].shape[1]], experts=np.transpose(experts_high[:,k,:]), model=agg, loss_gradient=True, loss_type={'name':'pinball', 'tau':1-alpha/2})
        experts_low_pred[k,:] = mlpol_grad_low['prediction']
        experts_high_pred[k,:] = mlpol_grad_high['prediction']

    return {'Y_inf':experts_low_pred, 'Y_sup':experts_high_pred}