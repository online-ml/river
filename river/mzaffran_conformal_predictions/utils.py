import files
from scipy.stats import norm
import numpy as np

def compute_PI_metrics(method, n, train_size, n_rep, parent_results='results', parent_data='data', online=True, randomized=False, regression='Friedman', noise='ARMA', params_reg={}, params_noise={}, impute=False):

    name_dir, name_method = files.get_name_results(method, n, online=online, randomized=randomized, regression=regression, noise=noise,
                                                   params_noise=params_noise)
    results = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')

    assert results['Y_inf'].shape[0] >= n_rep, 'You have not run enough experiments, %d repetitions required, only %d realized.' %(n_rep, results['Y_inf'].shape[0])

    name_data = files.get_name_data(n, regression=regression, noise=noise, params_noise=params_noise, seed=n_rep)
    data = files.load_file(parent_data, name_data, 'pkl')

    contains = (data['Y'][:,train_size:] <= results['Y_sup'][:n_rep,:]) & (data['Y'][:,train_size:] >= results['Y_inf'][:n_rep,:])

    if impute and (method[:3] in ['ACP','Agg']):
        # Get reference to obtain y_chap
        name_dir, name_method = files.get_name_results('ACP_0', n, online=online, randomized=randomized, regression=regression, noise=noise,
                                                       params_noise=params_noise)
        results_ref = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')

        assert results_ref['Y_inf'].shape[0] >= n_rep, 'You have not run enough experiments, %d repetitions required, only %d realized.' %(n_rep, results['Y_inf'].shape[0])

        borne_sup = results_ref['Y_sup'][:n_rep,:]
        borne_inf = results_ref['Y_inf'][:n_rep,:]
        y_chap = (borne_sup+borne_inf)/2
        abs_res = np.abs(data['Y'][:n_rep,train_size:] - y_chap)
        max_eps = np.max(abs_res)
        val_max = y_chap+max_eps
        val_min = y_chap-max_eps

        borne_sup = results['Y_sup'][:n_rep,:]
        borne_inf = results['Y_inf'][:n_rep,:]
        borne_sup[np.isinf(borne_sup)] = val_max[np.isinf(borne_sup)]
        borne_inf[np.isinf(borne_inf)] = val_min[np.isinf(borne_inf)]
        borne_sup[borne_sup > val_max] = val_max[borne_sup > val_max]
        borne_inf[borne_inf < val_min] = val_min[borne_inf < val_min]

        lengths = borne_sup - borne_inf

    else:
        lengths = results['Y_sup'][:n_rep,:] - results['Y_inf'][:n_rep,:]
    #times = results['Time'][:n_rep,:]

    return contains, lengths#, times

def compute_true_length(alpha, noise='ARMA', params_noise={}, horizon='Infinite'):

    assert noise in ['ARMA', 'Gaussian'], 'noise must be either ARMA or Gaussian'
    ar = params_noise['ar']
    ma = params_noise['ma']
    p = len(ar)
    q = len(ma)
    if (p==2) and (q==1):
        # then it is an AR(1)
        phi = -ar[1]
        theta = 0
    elif (p==1) and (q==2):
        # then it is an MA(1)
        phi = 0
        theta = ma[1]
    elif (p==2) and (q == 2):
        # then it is an ARMA(1,1)
        phi = -ar[1]
        theta = ma[1]
    elif (p==1) and (q == 1):
        # just a WN
        phi = 0
        theta = 0

    sum_squared_coef = (1+2*theta*phi+theta**2)/(1-phi**2)

    if 'process_variance' in params_noise:
        var = params_noise['process_variance']
        scale = np.sqrt(var/sum_squared_coef)
    else:
        if 'scale' in params_noise:
            scale = params_noise['scale']
        else:
            scale = 1
        var = (scale**2)*sum_squared_coef

    if horizon == 'Infinite':
        quantile = norm.ppf(1-alpha/2,scale=np.sqrt(var))
        length = 2*quantile
    return length

def gamma_opt_warm_up(tab_gamma,alpha,warm_up,n,train_size,n_rep,regression,noise,params_noise,parent_data='data',parent_results='results'):

    test_size = n - train_size

    methods = []
    for gamma in tab_gamma:
        methods.append('ACP_'+str(gamma))

    contains = np.empty((len(tab_gamma),n_rep,test_size))
    lengths = np.empty((len(tab_gamma),n_rep,test_size))
    y_sup = np.empty((len(tab_gamma),n_rep,test_size))
    y_inf = np.empty((len(tab_gamma),n_rep,test_size))

    name = files.get_name_data(n, regression=regression, noise=noise, params_noise=params_noise, seed=n_rep)
    data = files.load_file(parent_data, name, 'pkl')

    for k,method in enumerate(methods):
        name_dir, name_method = files.get_name_results(method, n, regression=regression, noise=noise,
                                                       params_noise=params_noise)
        results = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')
        contains[k,:,:] = (data['Y'][:n_rep,train_size:] <= results['Y_sup'][:n_rep,:]) & (data['Y'][:n_rep,train_size:] >= results['Y_inf'][:n_rep,:])
        lengths[k,:,:] = results['Y_sup'][:n_rep,:] - results['Y_inf'][:n_rep,:]
        y_sup[k,:,:] = results['Y_sup'][:n_rep,:]
        y_inf[k,:,:] = results['Y_inf'][:n_rep,:]

    contains_opt = np.empty((n_rep, test_size))
    lengths_opt = np.empty((n_rep, test_size))
    gammas_opt = np.empty((n_rep, test_size))
    y_sup_opt = np.empty((n_rep, test_size))
    y_inf_opt = np.empty((n_rep, test_size))
    for k in range(n_rep):
        contains_opt[k,0] = contains[0,k,0]
        lengths_opt[k,0] = lengths[0,k,0]
        y_sup_opt[k,0] = y_sup[0,k,0]
        y_inf_opt[k,0] = y_inf[0,k,0]
        gammas_opt[k,0] = 0
        for i in range(test_size-1):
            if i > warm_up :
                mean_cov = 1-np.mean(contains[:,k,:(i+1)],axis=1)
                mean_len = np.mean(lengths[:,k,:(i+1)],axis=1)
                mask = (mean_cov >= (1 - alpha))
                if True in mask:
                    best_idg = int(np.argwhere(mask)[np.argmin(mean_len[mask])])
                else:
                    mae = np.abs(mean_cov - (1-alpha))
                    minimizers = list(np.where(mae == np.min(mae))[0])
                    if len(minimizers) == 1:
                        best_idg = int(np.argmin(mae))
                    else:
                        mask_mae = (mae == np.min(mae))
                        best_idg = int(np.argwhere(mask_mae)[np.argmin(mean_len[mask_mae])])
                contains_opt[k,(i+1)] = contains[best_idg,k,(i+1)]
                lengths_opt[k,(i+1)] = lengths[best_idg,k,(i+1)]
                y_sup_opt[k,(i+1)] = y_sup[best_idg,k,(i+1)]
                y_inf_opt[k,(i+1)] = y_inf[best_idg,k,(i+1)]
                gammas_opt[k,(i+1)] = tab_gamma[best_idg]
            else :
                contains_opt[k,(i+1)] = contains[0,k,(i+1)]
                lengths_opt[k,(i+1)] = lengths[0,k,(i+1)]
                y_sup_opt[k,(i+1)] = y_sup[0,k,(i+1)]
                y_inf_opt[k,(i+1)] = y_inf[0,k,(i+1)]
                gammas_opt[k,(i+1)] = 0

    results_opt = {'contains': contains_opt, 'lengths': lengths_opt, 'gammas': gammas_opt,
                   'Y_sup':y_sup_opt,'Y_inf':y_inf_opt}
    return results_opt
