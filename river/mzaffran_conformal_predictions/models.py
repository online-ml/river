import files

import os
import time
from tqdm.autonotebook import tqdm
import numpy as np
np.warnings.filterwarnings('ignore')

from scipy.stats import norm

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

## EnbPI
from enbpi.PI_class_EnbPI import prediction_interval
import enbpi.utils_EnbPI as util

def fit_predict(X, Y, alpha, methods, params_methods, basemodel, params_basemodel, train_size):
    """ Function that trains the methods provided on samples from X,Y data and
    use these trained methods to predict on the last samples.

    Parameters
    ----------

    X : covariates, array of dim d x n (d number of covariates, n number of observations/sample size)
    Y : response, array of dim n x 1
    alpha : miscoverage level, in [0,1]
    methods : list of strings, containing the methods to apply
    params_methods :
    basemodel : regression basemodel, currently can be 'RF' or 'OLS'
    params_basemodel :
    train_size : number of samples to keep for training only
                (the first train_size samples will be kept, and the methods will preedict on the last ones)

    Returns
    -------

    y_lowers : inferior bound of the predicted intervals, array of dim len(methods) x (n-train_size)
               results are in the same order than the methods list
    y_uppers : superior bound of the predicted intervals, array of dim len(methods) x (n-train_size)
    times : user time spent for each prediction (proper training, calibration and prediction)
    times_proc : CPU time spent for each prediction (proper training, calibration and prediction)

    """

    # basics parameters
    n = len(Y)
    test_size = n - train_size

    # randomized will choose randomly which sample are for training and which are for calibration
    if 'randomized' in params_methods:
        randomized = params_methods['randomized']
    else:
        randomized = False

    # creation of the id for training and calibration
    if randomized:
        idx = np.random.permutation(train_size)
    else:
        idx = np.array(range(train_size))
    n_half = int(np.floor(train_size/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

    # initialization of the arrays containing the bounds of the intervals, and the computation times
    y_lowers = np.empty((len(methods),test_size))
    y_uppers = np.empty((len(methods),test_size))
    times = np.empty((len(methods),test_size))
    times_proc = np.empty((len(methods),test_size))

    # online will retrain the methods for each new test point, while offline will train
    # once on the first training set and use this model to predict on the whole test set
    if 'online' in  params_methods:
        online = params_methods['online']
    else:
        online = True

    # methods definitions
    EnbPIs = ['EnbPI','EnbPI_Mean']
    mean_reg_methods = ['Gaussian', 'CP', 'ACP']

    if any(method in methods for method in mean_reg_methods):
        mean_reg = True
    else:
        mean_reg = False

    # retrieve the basemodel parameters
    assert basemodel in ['RF','OLS'], 'basemodel must be RF or OLS.'
    if basemodel == 'RF':
        cores = params_basemodel['cores']
        if mean_reg:
            n_estimators = params_basemodel['n_estimators']
            min_samples_leaf = params_basemodel['min_samples_leaf']
            max_features = params_basemodel['max_features']

    # initialize parameters for 'ACP'
    if 'ACP' in methods:
        alpha_t = alpha
        gamma = params_methods['gamma']

    if online:

        for i in range(test_size):
            # define the new training and testing set.
            # the training set will then be splitted onto train/calibration
            # the testing set is just the i-th point
            x_train = np.transpose(X)[i:(train_size+i),]
            x_test = np.transpose(X)[(train_size+i),].reshape(1, -1)
            y_train = Y[i:(train_size+i)]
            y_test = Y[(train_size+i)]

            if mean_reg:
                if basemodel == 'RF':
                    # define RF model
                    reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                random_state=1, n_jobs=cores)
                elif basemodel == 'OLS':
                    reg = LinearRegression()

                start_mean_reg = time.time()
                start_mean_reg_proc = time.process_time()

                # fit the underlying model on the proper training points (subset of training points)
                reg.fit(x_train[idx_train,:], y_train[idx_train])

                # calibration predictions (subset of training points)
                y_pred_cal = reg.predict(x_train[idx_cal,:])

                # compute the calibrated residuals
                res_cal = y_train[idx_cal]-y_pred_cal

                # predict on the test point
                y_pred = reg.predict(x_test)
                end_mean_reg_proc = time.process_time()
                end_mean_reg = time.time()

            for idm, method in enumerate(methods):

                if method == "Gaussian":
                    start_gaussian = time.time()
                    start_gaussian_proc = time.process_time()
                    # create the bounds for the gaussian interval, centered around y_pred
                    window = norm.ppf(1-alpha/2)*np.std(res_cal)
                    y_lower_i, y_upper_i = y_pred-window, y_pred+window
                    end_gaussian_proc = time.process_time()
                    end_gaussian = time.time()
                    time_method = end_mean_reg-start_mean_reg+end_gaussian-start_gaussian
                    time_method_proc = end_mean_reg_proc-start_mean_reg_proc+end_gaussian_proc-start_gaussian_proc

                elif method == "CP":
                    start_cp = time.time()
                    start_cp_proc = time.process_time()
                    # compute the score (ie absolute value of the residuals)
                    res_cal_cp = np.abs(res_cal)
                    # compute the corrected quantile
                    window = np.quantile(res_cal_cp,(1-alpha)*(1+1/len(idx_cal)))
                    # create the bounds for the CP interval, centered around y_pred
                    y_lower_i, y_upper_i = y_pred-window, y_pred+window
                    end_cp_proc = time.process_time()
                    end_cp = time.time()
                    time_method = end_mean_reg-start_mean_reg+end_cp-start_cp
                    time_method_proc = end_mean_reg_proc-start_mean_reg_proc+end_cp_proc-start_cp_proc

                elif method == "ACP":
                    start_acp = time.time()
                    start_acp_proc = time.process_time()
                    # compute the score (ie absolute value of the residuals)
                    res_cal_acp = np.abs(res_cal)
                    if(alpha_t >= 1): # => 1-alpha_t <= 0 => predict empty set
                        y_lower_i, y_upper_i = 0, 0
                        err = 1 # err = 1 if the point is not included, 0 otherwise
                    elif(alpha_t <= 0): # => 1-alpha_t >= 1 => predict the whole real line
                        y_lower_i, y_upper_i = -np.inf, np.inf
                        err = 0
                    else: # => 1-alpha_t in ]0,1[ => compute the quantiles
                        # compute the updated quantile
                        window = np.quantile(res_cal_acp,(1-alpha_t))
                        # create the bounds for the ACP interval, centered around y_pred
                        y_lower_i, y_upper_i = y_pred-window, y_pred+window
                        err = 1-float((y_lower_i <= Y[train_size+i]) & (Y[train_size+i] <= y_upper_i))
                    # compute next value of alpha_t using updating scheme
                    alpha_t = alpha_t + gamma*(alpha-err)
                    end_acp_proc = time.process_time()
                    end_acp = time.time()
                    time_method = end_mean_reg-start_mean_reg+end_acp-start_acp
                    time_method_proc = end_mean_reg_proc-start_mean_reg_proc+end_acp_proc-start_acp_proc

                if method not in EnbPIs:
                    # save the results in the array of results for each method
                    y_lowers[idm,i] = float(y_lower_i)
                    y_uppers[idm,i] = float(y_upper_i)
                    times[idm,i] = time_method
                    times_proc[idm,i] = time_method_proc

    else: # if offline
        # define the new training and testing set.
        # the training set will then be splitted onto train/calibration
        x_train = np.transpose(X)[:train_size,]
        x_test = np.transpose(X)[train_size:,]
        y_train = Y[:train_size]
        y_test = np.array([Y[train_size:]])

        if mean_reg:
            if basemodel == 'RF':
                # define RF model
                reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                            random_state=1,n_jobs=cores)
            elif basemodel == 'OLS':
                reg = LinearRegression()

            start_mean_reg = time.time()
            start_mean_reg_proc = time.process_time()

            # fit the underlying model on the proper training points (subset of training points)
            reg.fit(x_train[idx_train,:], y_train[idx_train])

            # calibration predictions (subset of training points)
            y_pred_cal = reg.predict(x_train[idx_cal,:])

            # compute the calibrated residuals
            res_cal = y_train[idx_cal]-y_pred_cal

            # predict on the test point
            y_pred = reg.predict(x_test)

            end_mean_reg_proc = time.process_time()
            end_mean_reg = time.time()

        for idm, method in enumerate(methods):

            if method == "Gaussian":
                start_gaussian = time.time()
                start_gaussian_proc = time.process_time()
                window = norm.ppf(1-alpha/2)*np.std(res_cal)
                y_lower_i, y_upper_i = y_pred-window, y_pred+window
                end_gaussian_proc = time.process_time()
                end_gaussian = time.time()
                time_method = end_mean_reg-start_mean_reg+end_gaussian-start_gaussian
                time_method_proc = end_mean_reg_proc-start_mean_reg_proc+end_gaussian_proc-start_gaussian_proc

            elif method == "CP":
                start_cp = time.time()
                start_cp_proc = time.process_time()
                res_cal_cp = np.abs(res_cal)
                window = np.quantile(res_cal_cp,(1-alpha)*(1+1/len(idx_cal)))
                y_lower_i, y_upper_i = y_pred-window, y_pred+window
                end_cp_proc = time.process_time()
                end_cp = time.time()
                time_method = end_mean_reg-start_mean_reg+end_cp-start_cp
                time_method_proc = end_mean_reg_proc-start_mean_reg_proc+end_cp_proc-start_cp_proc

            if method not in EnbPIs:
                y_lowers[idm,:] = y_lower_i
                y_uppers[idm,:] = y_upper_i
                times[idm,:] = time_method
                times_proc[idm,:] = time_method_proc

    # create list of EnbPI methods to run (ie intersection of methods to run and EnbPIs methods)
    EnbPIs_to_run = list(set(methods) & set(EnbPIs))
    if len(EnbPIs_to_run) > 0: # if at least one EnbPI method to run
        for method in EnbPIs_to_run:
            idm = methods.index(method)

            # standard EnbPI parameters
            methods = ['Ensemble']
            itrial = 1
            miss_test_idx = []
            stride = 1
            data_name = ['Friedman_ARMA_Simulations']

            B = params_methods['B']
            if 'mean' in params_methods:
                mean = params_methods['mean']
            else:
                mean = False

            n_estimators = params_basemodel['n_estimators']
            min_samples_leaf = params_basemodel['min_samples_leaf']
            max_features = params_basemodel['max_features']

            x_train = np.transpose(X)[:train_size,]
            x_predict = np.transpose(X)[train_size:,]
            y_train = Y[:train_size]
            y_predict = np.array([Y[train_size:]])

            random_forest = RandomForestRegressor(n_estimators=n_estimators, criterion='mse', random_state=1,
                                                  bootstrap=False, min_samples_leaf=min_samples_leaf, max_features=max_features)

            start_enbpi = time.time()
            start_enbpi_proc = time.process_time()
            rf_results = prediction_interval(random_forest,  x_train, x_predict, y_train, y_predict)
            result_rf = rf_results.run_experiments(alpha, B, stride, data_name, itrial, miss_test_idx, methods=methods,
                                                   get_plots=True, mean=mean)
            end_enbpi_proc = time.process_time()
            end_enbpi = time.time()

            y_lowers[idm,:] = result_rf[0]['lower']
            y_uppers[idm,:] = result_rf[0]['upper']
            times[idm,:] = end_enbpi - start_enbpi
            times_proc[idm,:] = end_enbpi_proc - start_enbpi_proc

    return y_lowers, y_uppers, times, times_proc

def fit_predict_ACPs(X, Y, alpha, tab_gamma, basemodel, params_basemodel, train_size):

    n = len(Y)
    test_size = n - train_size
    idx = np.array(range(train_size))
    n_half = int(np.floor(train_size/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

    y_lowers = np.empty((len(tab_gamma),test_size))
    y_uppers = np.empty((len(tab_gamma),test_size))
    gammas = np.empty((len(tab_gamma),test_size))
    tab_alpha_t = np.full((len(tab_gamma),test_size), alpha)

    tab_err_gamma = np.empty((len(tab_gamma),test_size))
    tab_len_gamma = np.empty((len(tab_gamma),test_size))

    # methods and parameters
    mean_reg = True

    assert basemodel in ['RF','OLS'], 'basemodel must be RF or OLS.'
    if basemodel == 'RF':
        if mean_reg:
            n_estimators = params_basemodel['n_estimators']
            min_samples_leaf = params_basemodel['min_samples_leaf']
            max_features = params_basemodel['max_features']

    for i in range(test_size):
        x_train = np.transpose(X)[i:(train_size+i),]
        x_test = np.transpose(X)[(train_size+i),].reshape(1, -1)
        y_train = Y[i:(train_size+i)]
        y_test = Y[(train_size+i)]

        if mean_reg:
            if basemodel == 'RF':
                # define RF model
                reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                            random_state=1)
            elif basemodel == 'OLS':
                reg = LinearRegression()

            reg.fit(x_train[idx_train,:], y_train[idx_train])

            # calibration predictions
            y_pred_cal = reg.predict(x_train[idx_cal,:])
            res_cal = np.abs(y_train[idx_cal]-y_pred_cal)

            y_pred = reg.predict(x_test)

        for idg, gamma in enumerate(tab_gamma):
            alpha_t = tab_alpha_t[idg,i]
            # Original ACP
            if(1-alpha_t <= 0):
                y_lower_i, y_upper_i = 0, 0
                err = 1
            elif(1-alpha_t >= 1):
                y_lower_i, y_upper_i = -np.inf, np.inf
                err = 0
            else:
                window = np.quantile(res_cal,1-alpha_t)
                y_lower_i, y_upper_i = y_pred-window, y_pred+window
                err = 1-float((y_lower_i <= Y[train_size+i]) & (Y[train_size+i] <= y_upper_i))
            y_lowers[idg,i] = float(y_lower_i)
            y_uppers[idg,i] = float(y_upper_i)
            gammas[idg,i] = float(gamma)
            alpha_t = alpha_t + gamma*(alpha-err)
            if i < (test_size-1):
                tab_alpha_t[idg,i+1] = alpha_t
            tab_err_gamma[idg,i] = err
            tab_len_gamma[idg,i] = y_upper_i - y_lower_i

    return y_lowers, y_uppers, tab_alpha_t, gammas

def run_experiments(data, alpha, methods, params_methods, basemodel, params_basemodel,
                    n_rep, regression, noise, params_reg, params_noise, train_size,
                    parent_results='results'):

    if 'EnbPI' in methods:
        if 'mean' in params_methods:
            if params_methods['mean']:
                name_enbpi = 'EnbPI_Mean'
                methods[methods.index('EnbPI')] = name_enbpi
    results_methods = dict.fromkeys(methods)
    n = data['Y'].shape[1]

    if 'online' in params_methods:
        online = params_methods['online']
    else:
        online = True

    if 'randomized' in params_methods:
        randomized = params_methods['randomized']
    else:
        randomized = False

    for method in methods:
        name_dir, name_method = files.get_name_results(method, n, online, randomized, regression=regression, noise=noise,
                                                       params_noise=params_noise)
        if not os.path.isdir(parent_results+'/'+name_dir):
            os.mkdir(parent_results+'/'+name_dir)
        elif os.path.isfile(parent_results+'/'+name_dir+'/'+name_method+'.pkl'):
            results = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')
            results_methods[method] = results

    all_methods_ran = []

    for k in tqdm(range(n_rep)):

        methods_to_run = []

        for method in results_methods:
            if results_methods[method] is None:
                methods_to_run.append(method)
            elif results_methods[method]['Y_inf'].shape[0]-1 < k:
                methods_to_run.append(method)

        X = data['X'][k,:,:]
        Y = data['Y'][k,:]

        y_lowers, y_uppers, times, times_proc = fit_predict(X, Y, alpha, methods_to_run, params_methods, basemodel, params_basemodel, train_size)

        for idm, method in enumerate(methods_to_run):
            results = results_methods[method]
            if results is None:
                results = {'Y_inf': np.array([y_lowers[idm,:]]), 'Y_sup': np.array([y_uppers[idm,:]]),
                           'Time': np.array([times[idm,:]]), 'Time_CPU': np.array([times_proc[idm,:]])}
            else:
                results['Y_inf'] = np.vstack((results['Y_inf'],np.array([y_lowers[idm,:]])))
                results['Y_sup'] = np.vstack((results['Y_sup'],np.array([y_uppers[idm,:]])))
                results['Time'] = np.vstack((results['Time'],np.array([times[idm,:]])))
                results['Time_CPU'] = np.vstack((results['Time_CPU'],np.array([times_proc[idm,:]])))
            results_methods[method] = results

        all_methods_ran = np.append(all_methods_ran, methods_to_run)

    all_methods_ran = np.unique(all_methods_ran)
    return results_methods, all_methods_ran

def run_multiple_gamma_ACP(data, alpha, tab_gamma, basemodel, params_basemodel,
                           n_rep, regression, noise, params_reg, params_noise, train_size,
                           parent_results='results'):

   #assert len(tab_gamma)>1, 'tab_gamma should contain multiple values for gamma.'

   methods = []
   for gamma in tab_gamma:
       methods.append('ACP_'+str(gamma))

   results_methods = dict.fromkeys(methods)
   n = data['Y'].shape[1]

   for method in methods:
       name_dir, name_method = files.get_name_results(method, n, True, regression=regression, noise=noise,
                                                      params_noise=params_noise)
       if not os.path.isdir(parent_results+'/'+name_dir):
           os.mkdir(parent_results+'/'+name_dir)

   for k in tqdm(range(n_rep)):

       X = data['X'][k,:,:]
       Y = data['Y'][k,:]

       y_lowers, y_uppers, alpha_t, gammas = fit_predict_ACPs(X, Y, alpha, tab_gamma, basemodel, params_basemodel, train_size)

       for idm, method in enumerate(methods):
           # methods contain ACP_gamma in the same order than tab_gamma, and then eventually ACP_optimized
           results = results_methods[method]
           if results is None:
               results = {'Y_inf': np.array([y_lowers[idm,:]]), 'Y_sup': np.array([y_uppers[idm,:]]),
                          'alpha_t': np.array([alpha_t[idm,:]]), 'gammas': np.array([gammas[idm,:]])}
           else:
               results['Y_inf'] = np.vstack((results['Y_inf'],np.array([y_lowers[idm,:]])))
               results['Y_sup'] = np.vstack((results['Y_sup'],np.array([y_uppers[idm,:]])))
               results['alpha_t'] = np.vstack((results['alpha_t'],np.array([alpha_t[idm,:]])))
               results['gammas'] = np.vstack((results['gammas'],np.array([gammas[idm,:]])))
           results_methods[method] = results

   return results_methods, methods

def run_experiments_real_data(data, alpha, methods, params_methods, basemodel, params_basemodel, train_size, dataset,
                              erase=False, parent_results='results'):

    if 'EnbPI' in methods:
        if 'mean' in params_methods:
            if params_methods['mean']:
                name_enbpi = 'EnbPI_Mean'
                methods[methods.index('EnbPI')] = name_enbpi
    results_methods = dict.fromkeys(methods)

    if 'online' in params_methods:
        online = params_methods['online']
    else:
        online = True

    for method in methods:
        name_dir, name_method = files.get_name_results(method, online=online, dataset=dataset, basemodel=basemodel)
        if not os.path.isdir(parent_results+'/'+name_dir):
            os.mkdir(parent_results+'/'+name_dir)
        elif os.path.isfile(parent_results+'/'+name_dir+'/'+name_method+'.pkl') and not erase:
            results = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')
            results_methods[method] = results

    all_methods_ran = []

    methods_to_run = []

    for method in results_methods:
        if results_methods[method] is None:
            methods_to_run.append(method)

    X = data['X']
    Y = data['Y']

    y_lowers, y_uppers, times, times_proc = fit_predict(X, Y, alpha, methods_to_run, params_methods, basemodel, params_basemodel, train_size)

    for idm, method in enumerate(methods_to_run):
        results = results_methods[method]
        if results is None:
            results = {'Y_inf': np.array([y_lowers[idm,:]]), 'Y_sup': np.array([y_uppers[idm,:]]),
                       'Time': np.array([times[idm,:]]), 'Time_CPU': np.array([times_proc[idm,:]])}
        else:
            results['Y_inf'] = np.vstack((results['Y_inf'],np.array([y_lowers[idm,:]])))
            results['Y_sup'] = np.vstack((results['Y_sup'],np.array([y_uppers[idm,:]])))
            results['Time'] = np.vstack((results['Time'],np.array([times[idm,:]])))
            results['Time_CPU'] = np.vstack((results['Time_CPU'],np.array([times_proc[idm,:]])))
        results_methods[method] = results

    all_methods_ran = np.append(all_methods_ran, methods_to_run)

    all_methods_ran = np.unique(all_methods_ran)
    return results_methods, all_methods_ran

def run_multiple_gamma_ACP_real_data(data, alpha, tab_gamma, basemodel, params_basemodel,
                                     train_size, dataset, erase=False, parent_results='results'):

   #assert len(tab_gamma)>1, 'tab_gamma should contain multiple values for gamma.'

   methods = []
   for gamma in tab_gamma:
       methods.append('ACP_'+str(gamma))

   results_methods = dict.fromkeys(methods)

   for method in methods:
       name_dir, name_method = files.get_name_results(method, dataset=dataset, basemodel=basemodel)
       if not os.path.isdir(parent_results+'/'+name_dir):
           os.mkdir(parent_results+'/'+name_dir)
       elif os.path.isfile(parent_results+'/'+name_dir+'/'+name_method+'.pkl') and not erase:
           results = files.load_file(parent_results+'/'+name_dir, name_method, 'pkl')
           results_methods[method] = results

   all_methods_ran = []

   methods_to_run = []
   tab_gamma_to_run = []

   for idg,gamma in enumerate(tab_gamma):
       method = methods[idg]
       if results_methods[method] is None and not erase:
           methods_to_run.append(method)
           tab_gamma_to_run.append(gamma)
       elif erase:
           methods_to_run.append(method)
           tab_gamma_to_run.append(gamma)

   X = data['X']
   Y = data['Y']

   if len(tab_gamma_to_run) > 0:
       y_lowers, y_uppers, alpha_t, gammas = fit_predict_ACPs(X, Y, alpha, tab_gamma_to_run, basemodel, params_basemodel, train_size)

   for idm, method in enumerate(methods_to_run):
       # methods contain ACP_gamma in the same order than tab_gamma, and then eventually ACP_optimized
       results = results_methods[method]
       if results is None:
           results = {'Y_inf': np.array([y_lowers[idm,:]]), 'Y_sup': np.array([y_uppers[idm,:]]),
                      'alpha_t': np.array([alpha_t[idm,:]]), 'gammas': np.array([gammas[idm,:]])}
       else:
           results['Y_inf'] = np.vstack((results['Y_inf'],np.array([y_lowers[idm,:]])))
           results['Y_sup'] = np.vstack((results['Y_sup'],np.array([y_uppers[idm,:]])))
           results['alpha_t'] = np.vstack((results['alpha_t'],np.array([alpha_t[idm,:]])))
           results['gammas'] = np.vstack((results['gammas'],np.array([gammas[idm,:]])))
       results_methods[method] = results

   return results_methods, methods
