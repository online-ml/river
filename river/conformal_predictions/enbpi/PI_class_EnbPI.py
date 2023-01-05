import importlib
import warnings
from enbpi.utils_EnbPI import generate_bootstrap_samples, strided_app, weighted_quantile
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.optimizers import Adam
import sys
warnings.filterwarnings("ignore")


class prediction_interval():
    '''
        Create prediction intervals using different methods (Ensemble, LOO, ICP, weighted...)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        '''
            Fit_func: ridge, lasso, linear model, data
        '''
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # it will be updated with a list of bootstrap models, fitted on subsets of training data
        self.Ensemble_fitted_func = []
        # it will store residuals e_1, e_2,... from Ensemble
        self.Ensemble_online_resid = np.array([])
        self.ICP_fitted_func = []  # it only store 1 fitted ICP func.
        # it will store residuals e_1, e_2,... from ICP
        self.ICP_online_resid = np.array([])
        self.WeightCP_online_resid = np.array([])
    '''
        Algorithm: Ensemble (online)
            Main difference from earlier is
            1. We need to store these bootstrap estimators f^b
            2. when aggregating these stored f^b to make prediction on future points,
            do not aggregate all of them but randomly select B*~Binom(B,e^-1 ~= (1-1/k)^k) many f^b
            3. the residuals grow in length, so that a new point uses all previous residuals to create intervals
            (Thus intervals only get wider, not shorter)
    '''

    def fit_bootstrap_models_online(self, alpha, B, miss_test_idx, mean=False):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train) and calculate predictions on original data X_train
          Return 1-\alpha quantile of each prdiction on self.X_predict, also
          1. update self.Ensemble_fitted_func with bootstrap estimators and
          2. update self.Ensemble_online_resid with LOO online residuals (from training)
          Update:
           Include tilt option (only difference is using a different test data, so just chaneg name from predict to predict_tilt)
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = generate_bootstrap_samples(n, n, B)
        # hold predictions from each f^b
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict = np.zeros((n, n1))
        ind_q = int((1-alpha)*n)
        #print('ind_q:',ind_q)
        #print('n:',n)
        for b in range(B):
            model = self.regressor
            if self.regressor.__class__.__name__ == 'Sequential':
                #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                if self.regressor.name == 'NeuralNet':
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
                else:
                    # This is RNN, mainly have different shape and decrease epochs
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            else:
                model = model.fit(self.X_train[boot_samples_idx[b], :],
                                  self.Y_train[boot_samples_idx[b], ])
            boot_predictions[b] = model.predict(np.r_[self.X_train, self.X_predict]).flatten()
            self.Ensemble_fitted_func.append(model)
            in_boot_sample[b, boot_samples_idx[b]] = True
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                resid_LOO = np.abs(self.Y_train[i] - boot_predictions[b_keep, i].mean())
                self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                resid_LOO = np.abs(self.Y_train[i])
                self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
                out_sample_predict[i] = np.zeros(n1)
        if not mean:
            sorted_out_sample_predict = np.sort(out_sample_predict, axis=0)[ind_q]  # length n1
        # HERE: modification
        if mean:
            sorted_out_sample_predict = np.mean(out_sample_predict, axis=0)
        # TODO: Change this, because ONLY minus the non-missing predictions
        # However, need to make sure same length is maintained, o/w sliding cause problem

        #print('out_sample:',out_sample_predict)
        #print('out_sample shape:',out_sample_predict.shape)
        #print('sorted:', np.sort(out_sample_predict, axis=0))
        #print('sorted shape:',np.sort(out_sample_predict, axis=0).shape)
        #print('sorted_selected:',sorted_out_sample_predict)
        #print('sorted_selected shape:',sorted_out_sample_predict.shape)
        #print('Y_predict:', self.Y_predict)
        resid_out_sample = np.abs(sorted_out_sample_predict-self.Y_predict)
        if len(miss_test_idx) > 0:
            # Replace missing residuals with that from the immediate predecessor that is not missing
            for l in range(len(miss_test_idx)):
                i = miss_test_idx[l]
                if i > 0:
                    j = i-1
                    while j in miss_test_idx[:l]:
                        j -= 1
                    resid_out_sample[i] = resid_out_sample[j]

                else:
                    # The first Y during testing is missing, let it be the last of the training residuals
                    # note, training data already takes out missing values, so doing is is fine
                    resid_out_sample[0] = self.Ensemble_online_resid[-1]
        self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_out_sample)
        return(sorted_out_sample_predict)

    def compute_PIs_Ensemble_online(self, alpha, B, stride, miss_test_idx, density_est=False, mean=False):
        '''
            Note, this is not online version, so all test points have the same width
        '''
        n = len(self.X_train)
        n1 = len(self.Y_predict)
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.fit_bootstrap_models_online(
            alpha, B, miss_test_idx,mean)  # length of n1
        ind_q = int(100*(1-alpha))
        # start = time.time()
        if density_est:
            blocks = int(n1/stride)
            ind_q = np.zeros(blocks)
            p_vals = self.Ensemble_online_resid[:n]  # This will be changing
            p_vals = np.array([np.sum(i > p_vals)/len(p_vals) for i in p_vals])
            # Fill in first (block) of estimated quantiles:
            ind_q[0] = 100*beta_percentile(p_vals, alpha)
            # Fill in consecutive blocks
            for block in range(blocks-1):
                p_vals = p_vals[stride:]
                new_p_vals = self.Ensemble_online_resid[n+block*stride:n+(block+1)*stride]
                new_p_vals = np.array([np.sum(i > new_p_vals)/len(new_p_vals) for i in new_p_vals])
                p_vals = np.hstack((p_vals, new_p_vals))
                ind_q[block+1] = 100*beta_percentile(p_vals, alpha)
            ind_q = ind_q.astype(int)
            width = np.zeros(blocks)
            strided_resid = strided_app(self.Ensemble_online_resid[:-1], n, stride)
            for i in range(blocks):
                width[i] = np.percentile(strided_resid[i], ind_q[i], axis=-1)
        else:
            width = np.percentile(strided_app(
                self.Ensemble_online_resid[:-1], n, stride), ind_q, axis=-1)
        #print(self.Ensemble_online_resid)
        #print(self.Ensemble_online_resid[:-1])
        #print(strided_app(
    #        self.Ensemble_online_resid[:-1], n, stride))
    #    print(width)
        width = np.abs(np.repeat(width, stride))  # This is because |width|=T/stride.
    #    print(width)
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict-width,
                                          out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_Ensemble

    '''
        Jackknife+-after-bootstrap (used in Figure 6)
    '''

    def fit_bootstrap_models(self, B):
        '''
          Train B bootstrap estimators and calculate predictions on X_predict
          Return: list of matrices [M,P]
            samples_idx = B-by-m matrix, row b = indices of b-th bootstrap sample
            predictions = B-by-n1 matrix, row b = predictions from b-th bootstrap sample
              (n1=len(X_predict))
        '''
        n = len(self.X_train)
        boot_samples_idx = generate_bootstrap_samples(n, n, B)
        n1 = len(np.r_[self.X_train, self.X_predict])
        # P holds the predictions from individual bootstrap estimators
        predictions = np.zeros((B, n1), dtype=float)
        for b in range(B):
            model = self.regressor
            if self.regressor.__class__.__name__ == 'Sequential':
                #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                if self.regressor.name == 'NeuralNet':
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
                else:
                    # This is RNN, mainly have different shape and decrease epochs
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            else:
                model = model.fit(self.X_train[boot_samples_idx[b], :],
                                  self.Y_train[boot_samples_idx[b], ])
            predictions[b] = model.predict(np.r_[self.X_train, self.X_predict]).flatten()
        return([boot_samples_idx, predictions])

    def compute_PIs_JaB(self, alpha, B):
        '''
        Using mean aggregation
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        [boot_samples_idx, boot_predictions] = self.fit_bootstrap_models(B)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        for b in range(len(in_boot_sample)):
            in_boot_sample[b, boot_samples_idx[b]] = True
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n, n1))
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                resids_LOO[i] = np.abs(self.Y_train[i] - boot_predictions[b_keep, i].mean())
                muh_LOO_vals_testpoint[i] = boot_predictions[b_keep, n:].mean(0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                resids_LOO[i] = np.abs(self.Y_train[i])
                muh_LOO_vals_testpoint[i] = np.zeros(n1)
        ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
        return pd.DataFrame(
            np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO, axis=1).T[-ind_q],
                  np.sort(muh_LOO_vals_testpoint.T + resids_LOO, axis=1).T[ind_q-1]],
            columns=['lower', 'upper'])

    '''
        Inductive Conformal Prediction (online)
    '''

    def compute_PIs_ICP_online(self, alpha, l, density_est):
        '''Basic idea: Randomly subsample l data from X, fit a model on X, calculate residuals on all but the l data in (X,Y),
           and finally compute the CI using the quantiles
           Main difference from offline version:
            We also update the length of residuals so the interval widths only grow
           '''
        n = len(self.X_train)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            if self.regressor.name == 'NeuralNet':
                model.fit(self.X_train, self.Y_train,
                          epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            model.fit(X_train, Y_train,
                      epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(self.X_predict).flatten()
        self.ICP_online_resid = np.append(
            self.ICP_online_resid, calibrate_residuals)  # length n-l
        ind_q = int(100*(1-alpha))  # 1-alpha%
        width = np.abs(np.percentile(self.ICP_online_resid, ind_q, axis=-1).T)
        PIs_ICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                     out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_ICP

    '''
        Weighted Conformal Prediction
    '''

    def compute_PIs_Weighted_ICP_online(self, alpha, l, density_est):
        '''Basic idea: Randomly subsample l data from X, fit a model on X, calculate residuals on all but the l data in (X,Y),
           and finally compute the CI using the quantiles
           Caveat: the residuals are weighted by fitting a logistic regression on
           (X_calibrate, C=0) \cup (X_predict, C=1
           Main difference from offline version:
            We also update the length of residuals so the interval widths only grow
           '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # Main difference from ICP
        C_calibrate = np.zeros(n-l)
        C_predict = np.ones(n1)
        X_weight = np.r_[X_calibrate, self.X_predict]
        C_weight = np.r_[C_calibrate, C_predict]
        if len(X_weight.shape) > 2:
            # Reshape for RNN
            tot, _, shap = X_weight.shape
            X_weight = X_weight.reshape((tot, shap))
        clf = LogisticRegression(random_state=0).fit(X_weight, C_weight)
        Prob = clf.predict_proba(X_weight)
        Weights = Prob[:, 1]/(1-Prob[:, 0])  # n-l+n1 in length
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            if self.regressor.name == 'NeuralNet':
                model.fit(X_train, Y_train,
                          epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(self.X_predict).flatten()
        self.WeightCP_online_resid = np.append(
            self.WeightCP_online_resid, calibrate_residuals)  # length n-1
        width = np.abs(weighted_quantile(values=self.WeightCP_online_resid, quantiles=1-alpha,
                                         sample_weight=Weights[:n-l]))
        PIs_ICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                     out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_ICP

    def compute_PIs_ARIMA_online(self, alpha):
        '''
            Fit ARIMA(10,1,10) to all models
            Use train_size to form model and the rest to be out-sample-prediction
            return PI (in class, train_size would just be len(self.Y_train), data would be
            pd.DataFrame(np.r[self.Y_train,self.Y_predict]))
            Note, need to import statsmodels.api as sm
        '''
        # Concatenate training and testing together
        data = pd.DataFrame(np.r_[self.Y_train, self.Y_predict])
        # Train model
        train_size = len(self.Y_train)
        training_mod = sm.tsa.statespace.SARIMAX(data[:train_size], order=(10, 1, 10))
        print('training')
        training_res = training_mod.fit(disp=0)
        print('training done')
        # Use in full model
        mod = sm.tsa.SARIMAX(data, order=(10, 1, 10))
        res = mod.filter(training_res.params)
        # Get the insample prediction interval (which is outsample prediction interval)
        pred = res.get_prediction(start=data.index[train_size], end=data.index[-1])
        pred_int = pred.conf_int(alpha=alpha)  # prediction interval
        PIs_ARIMA = pd.DataFrame(
            np.c_[pred_int.iloc[:, 0], pred_int.iloc[:, 1]], columns=['lower', 'upper'])
        return(PIs_ARIMA)

    '''
        All together
    '''

    def run_experiments(self, alpha, B, stride, data_name, itrial,  miss_test_idx, true_Y_predict=[], density_est=False, get_plots=False, none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP'], mean=False):
        '''
            Note, it is assumed that real data has been loaded, so in actual execution,
            generate data before running this
            Default is for running real-data
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
        '''
        train_size = len(self.X_train)
        np.random.seed(98765+itrial)
        if none_CP:
            results = pd.DataFrame(columns=['itrial', 'dataname',
                                            'method', 'train_size', 'coverage', 'width'])
            print('Not using Conformal Prediction Methods')
            print('Running ARIMA(10,1,10)')
            PI_ARIMA = self.compute_PIs_ARIMA_online(alpha)
            coverage_ARIMA = ((np.array(PI_ARIMA['lower']) <= self.Y_predict) & (
                np.array(PI_ARIMA['upper']) >= self.Y_predict)).mean()
            print(f'Average Coverage is {coverage_ARIMA}')
            width_ARIMA = (PI_ARIMA['upper'] - PI_ARIMA['lower']).mean()
            print(f'Average Width is {width_ARIMA}')
            results.loc[len(results)] = [itrial, data_name, 'ARIMA',
                                         train_size, coverage_ARIMA, width_ARIMA]
        else:
            results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                            'method', 'train_size', 'coverage', 'width'])
            PIs = []
            for method in methods:
                print(f'Runnning {method}')
                if method == 'JaB':
                    B_ = B
                    n = len(self.X_train)
                    B = int(np.random.binomial(int(B_/(1-1./(1+train_size))**n),
                                               (1-1./(1+train_size))**n, size=1))
                    PI = self.compute_PIs_JaB(alpha, B)
                elif method == 'Ensemble':
                    PI = eval(f'compute_PIs_{method}_online({alpha},{B},{stride},{miss_test_idx},{density_est},{mean})',
                              globals(), {k: getattr(self, k) for k in dir(self)})
                else:
                    l = int(0.5*len(self.X_train))
                    PI = eval(f'compute_PIs_{method}_online({alpha},{l},{density_est})',
                              globals(), {k: getattr(self, k) for k in dir(self)})
                PIs.append(PI)
                coverage = ((np.array(PI['lower']) <= self.Y_predict) & (
                    np.array(PI['upper']) >= self.Y_predict)).mean()
                if len(true_Y_predict) > 0:
                    coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                        np.array(PI['upper']) >= true_Y_predict)).mean()
                print(f'Average Coverage is {coverage}')
                width = (PI['upper'] - PI['lower']).mean()
                print(f'Average Width is {width}')
                results.loc[len(results)] = [itrial, data_name,
                                             self.regressor.__class__.__name__, method, train_size, coverage, width]
        if get_plots:
            if none_CP:
                return([PI_ARIMA, results])
            else:
                PIs.append(results)
                '''Do 1,2,3 below with PIs_Ensemble and PI_ICP and (more) '''
                return(PIs)
        else:
            return(results)

    def series_vs_PI(self, PIs_ls, data_name, fit_func_name, one_dim=False):
        # names = {'RidgeCV': 'Ridge:', 'LassoCV': 'Lasso:',
        #          'RandomForestRegressor': "Random Forest:"}
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharey=True)
        n1 = len(self.Y_predict)
        axisfont = 20
        tickfont = 16
        plot_len = np.min([int(0.1*n1), 35])
        x_axis = np.arange(plot_len)  # just 0 to at most 35, it doesn't matter what x-axis is now
        ax[0].plot(x_axis, self.Y_predict[:plot_len],
                   color='black',  linestyle='dashed', label='Data', marker='o')
        ax[1].plot(x_axis, self.Y_predict[:plot_len],
                   color='black',  linestyle='dashed', label='Data', marker='o')
        color = ['blue', 'red', 'black']
        label_ls = ['Ensemble', 'ICP', 'Weighetd_ICP']
        for i in range(len(color)):
            # First multivariate, 0-2
            ax[0].plot(x_axis, PIs_ls[i]['lower'][:plot_len],
                       color=color[i], label=label_ls[i])
            ax[0].plot(x_axis, PIs_ls[i]['upper'][:plot_len], color=color[i])
            ax[0].set_xlabel('time', fontsize=axisfont)
            ax[0].tick_params(axis='both', which='major', labelsize=tickfont)
            # Then univariate, 3-5
            ax[1].plot(x_axis, PIs_ls[i+3]['lower'][:plot_len],
                       color=color[i], label=label_ls[i])
            ax[1].plot(x_axis, PIs_ls[i+3]['upper'][:plot_len], color=color[i])
            ax[1].set_xlabel('time', fontsize=axisfont)
            ax[1].tick_params(axis='both', which='major', labelsize=tickfont)
        ax[0].legend(fontsize=axisfont-2, loc='lower left')
        fig.tight_layout()
        plt.savefig(f'{data_name}_band_around_actual_{fit_func_name}.pdf', dpi=300, bbox_inches='tight',
                    pad_inches=0)
        plt.show()

    def Winkler_score(self, PIs_ls, data_name, methods_name, alpha, none_CP=False):
        # Examine if each test point is in the intervals
        # If in, then score += width of intervals
        # If not,
        # If true y falls under lower end, score += width + 2*(lower end-true y)/alpha
        # If true y lies above upper end, score += width + 2*(true y-upper end)/alpha
        n1 = len(self.Y_predict)
        score_ls = []
        if none_CP:
            score = 0
            for j in range(n1):
                upper = PIs_ls.loc[j, 'upper']
                lower = PIs_ls.loc[j, 'lower']
                width = upper-lower
                truth = self.Y_predict[j]
                if (truth >= lower) & (truth <= upper):
                    score += width
                elif truth < lower:
                    score += width + 2 * (lower-truth)/alpha
                else:
                    score += width + 2 * (truth-upper)/alpha
            score_ls.append(score)
        else:
            for i in range(len(methods_name)):
                score = 0
                for j in range(n1):
                    upper = PIs_ls[i].loc[j, 'upper']
                    lower = PIs_ls[i].loc[j, 'lower']
                    width = upper-lower
                    truth = self.Y_predict[j]
                    if (truth >= lower) & (truth <= upper):
                        score += width
                    elif truth < lower:
                        score += width + 2 * (lower-truth)/alpha
                    else:
                        score += width + 2 * (truth-upper)/alpha
                score_ls.append(score)
        return(score_ls)
