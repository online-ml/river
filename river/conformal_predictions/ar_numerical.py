import random
from tqdm.autonotebook import tqdm
import numpy as np
import statsmodels as sm
import statsmodels.tsa.arima_process
import utils
import pickle
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    default=0,
    type=int,
    nargs='?',
    const=0
)

args = parser.parse_args()
seed = args.seed

random.seed(seed)
np.random.seed(seed)

alpha = 0.1
n = 10
ncal = n
phis = [0,0.6,0.85,0.95,0.98,0.99,0.997,0.999]
gammas = np.linspace(0,0.2,100)

quantiles = np.empty((len(phis),len(gammas),ncal))
alphas = np.full((len(phis),len(gammas),ncal), alpha)
errs = np.full((len(phis),len(gammas),ncal), alpha)
epsilons = np.empty((len(phis),ncal))

for idp, phi in tqdm(enumerate(phis)):

    ar = [1,-phi]
    ma = [1]
    scale = 1
    scale_innov = np.sqrt((scale**2)*(1-phi**2))

    eps = sm.tsa.arima_process.arma_generate_sample(ar, ma, n, scale=scale_innov)
    epsilons[idp,:] = eps

    for idg, gamma in tqdm(enumerate(gammas)):

        alpha_t = alpha
        alpha_t_emp = alpha # inutile

        for i in range(ncal):

            alphas[idp,idg,i] = alpha_t

            if(1-alpha_t <= 0):
                quantile = 0
                err = True

            elif(1-alpha_t >= 1):
                quantile = np.inf
                err = False

            else:
                quantile = scipy.stats.norm.ppf(1-alpha_t/2, 0, scale)
                err = np.abs(eps[i]) > quantile

            quantiles[idp,idg,i] = quantile

            if err:
                alpha_t = alpha_t + gamma*(alpha-1)
            else:
                alpha_t = alpha_t + gamma*(alpha)

            errs[idp,idg,i] = 1*err

for i in range(len(phis)):
    quantiles[i,np.isinf(quantiles[i,:,:])] = np.max(np.abs(epsilons[i,:]))
    quantiles[i,quantiles[i,:,:] > np.max(np.abs(epsilons[i,:]))] = np.max(np.abs(epsilons[i,:]))

empirical_expectation_lengths_proj = 2*np.mean(quantiles, axis=2)
empirical_median_lengths_proj = 2*np.median(quantiles, axis=2)
results_stats = {'emp_exp_lengths_proj':empirical_expectation_lengths_proj,'emp_med_lengths_proj':empirical_median_lengths_proj,
'phis': phis, 'gammas': gammas}
name_outputs = 'results/ar_numerical/aci_theory_ar_fixed_var_1_proj_stats_all_imputed_seed%d.pkl'%seed
with open(name_outputs,'wb') as f:
    pickle.dump(results_stats, f)
