from scipy.linalg import norm
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from numpy.random import choice
import warnings
import os
import matplotlib.cm as cm
import time
import calendar
#from keras.layers import LSTM, Dense, Dropout
#from keras.models import Sequential
#from keras.optimizers import Adam
'''Helpers for read data '''


########################################################################################################################
########################################################################################################################
##############################################      utils_EnbPI.py     #################################################
########################################################################################################################
########################################################################################################################
def read_data(i, filename, max_data_size):
    if i == 0:
        '''
            All datasets are Multivariate time-series. They have respective Github for more details as well.
            1. Greenhouse Gas Observing Network Data Set
            Time from 5.10-7.31, 2010, with 4 samples everyday, 6 hours apart between data poits.
            Goal is to "use inverse methods to determine the optimal values of the weights in the weighted sum of 15 tracers that best matches the synthetic observations"
            In other words, find weights so that first 15 tracers will be as close to the last as possible.
            Note, data at many other grid cells are available. Others are in Downloads/🌟AISTATS Data/Greenhouse Data
            https://archive.ics.uci.edu/ml/datasets/Greenhouse+Gas+Observing+Network
        '''
        data = pd.read_csv(filename, header=None, sep=' ').T
        # data.shape  # 327, 16Note, rows are 16 time series (first 15 from tracers, last from synthetic).
    elif i == 1:
        '''
            2. Appliances energy prediction Data Set
            The data set is at 10 min for about 4.5 months.
            The column named 'Appliances' is the response. Other columns are predictors
            https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
        '''
        data = pd.read_csv(filename, delimiter=',')
        # data.shape  # (19736, 29)
        data.drop('date', inplace=True, axis=1)
        data.loc[:, data.columns != 'Appliances']
    elif i == 2:
        '''
            3. Beijing Multi-Site Air-Quality Data Data Set
            This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites.
            Time period from 3.1, 2013 to 2.28, 2017.
            PM2.5 or PM10 would be the response.
            https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
        '''
        data = pd.read_csv(filename)
        # data.shape  # 35064, 18
        # data.columns
        data.drop(columns=['No', 'year', 'month', 'day', 'hour',
                           'wd', 'station'], inplace=True, axis=1)
        data.dropna(inplace=True)
        # data.shape  # 32907, 11
        # data.head(5)
    else:
        """
            4 (Alternative). NREL Solar data at Atlanta Downtown in 2018. 24 observations per day and separately equally by 1H @ half an hour mark everytime
            Data descriptions see Solar Writeup
            Data download:
            (With API) https://nsrdb.nrel.gov/data-sets/api-instructions.html
            (Manual) https://maps.nrel.gov/nsrdb-viewer
        """
        data = pd.read_csv(filename, skiprows=2)
        # data.shape  # 8760, 14
        data.drop(columns=data.columns[0:5], inplace=True)
        data.drop(columns='Unnamed: 13', inplace=True)
        # data.shape  # 8760, 8
        # data.head(5)
    # pick maximum of X data points (for speed)
    data = data.iloc[:min(max_data_size, data.shape[0]), :]
    print(data.shape)
    return data

# Sec 8.3


def read_CA_data(filename):
    data = pd.read_csv(filename)
    # data.shape  # 8760, 14
    data.drop(columns=data.columns[0:6], inplace=True)
    return data


def read_wind_data():
    ''' Note, just use the 8760 hourly observation in 2019
    Github repo is here: https://github.com/Duvey314/austin-green-energy-predictor'''
    data_wind_19 = pd.read_csv('Data/Wind_Hackberry_Generation_2019_2020.csv')
    data_wind_19 = data_wind_19.iloc[:24*365, :]
    return data_wind_19


'''Helper for Multi-step ahead inference'''


def missing_data(data, missing_frac, update=False):
    n = len(data)
    idx = np.random.choice(n, size=int(missing_frac*n), replace=False)
    if update:
        data = np.delete(data, idx, 0)
    idx = idx.tolist()
    return (data, idx)


'''Neural Networks Regressors'''


# def keras_mod():
#     model = Sequential(name='NeuralNet')
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(1, activation='relu'))
#     # Compile model
#     opt = Adam(0.0005)
#     model.compile(loss='mean_squared_error', optimizer=opt)
#     return model


# def keras_rnn():
#     model = Sequential(name='RNN')
#     # For fast cuDNN implementation, activation = 'relu' does not work
#     model.add(LSTM(100, activation='tanh', return_sequences=True))
#     model.add(LSTM(100, activation='tanh'))
#     model.add(Dense(1, activation='relu'))
#     # Compile model
#     opt = Adam(0.0005)
#     model.compile(loss='mean_squared_error', optimizer=opt)
#     return model


'''Helper for ensemble'''


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def one_dimen_transform(Y_train, Y_predict, d):
    n = len(Y_train)
    n1 = len(Y_predict)
    X_train = np.zeros((n-d, d))  # from d+1,...,n
    X_predict = np.zeros((n1, d))  # from n-d,...,n+n1-d
    for i in range(n-d):
        X_train[i, :] = Y_train[i:i+d]
    for i in range(n1):
        if i < d:
            X_predict[i, :] = np.r_[Y_train[n-d+i:], Y_predict[:i]]
        else:
            X_predict[i, :] = Y_predict[i-d:i]
    Y_train = Y_train[d:]
    return([X_train, X_predict, Y_train, Y_predict])


'''Helper for doing online residual'''


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


'''Helper for Weighted ICP'''


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


"""
For Plotting results: width and coverage plots
"""


def plot_average_new(x_axis, x_axis_name, save=True, Dataname=['Solar_Atl'], two_rows=True):
    """Plot mean coverage and width for different PI methods and regressor combinations side by side,
       over rho or train_size or alpha_ls
       Parameters:
        data_type: simulated (2-by-3) or real data (2-by-2)
        x_axis: either list of train_size, or alpha
        x_axis_name: either train_size or alpha
    """
    ncol = 2
    Dataname.append(Dataname[0])  # for 1D results
    if two_rows:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    else:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    j = 0
    filename = {'alpha': 'alpha', 'train_size': 'train'}
    one_D = False
    for data_name in Dataname:
        # load appropriate data
        if j == 1 or one_D:
            results = pd.read_csv(f'Results/{data_name}_many_{filename[x_axis_name]}_new_1d.csv')
        else:
            results = pd.read_csv(f'Results/{data_name}_many_{filename[x_axis_name]}_new.csv')
        methods_name = np.unique(results.method)
        cov_together = []
        width_together = []
        # Loop through dataset name and plot average coverage and width for the particular regressor
        muh_fun = np.unique(results[results.method != 'ARIMA']
                            ['muh_fun'])  # First ARIMA, then Ensemble
        for method in methods_name:
            if method == 'ARIMA':
                results_method = results[(results['method'] == method)]
                if data_name == 'Network':
                    method_cov = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                else:
                    method_cov = results_method.groupby(
                        x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std']/np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std']/np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
            else:
                for fit_func in muh_fun:
                    results_method = results[(results['method'] == method) &
                                             (results['muh_fun'] == fit_func)]
                    if data_name == 'Network':
                        method_cov = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                    else:
                        method_cov = results_method.groupby(
                            x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std']/np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std']/np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
        # Plot
        # Parameters
        num_method = 1+len(muh_fun)  # ARIMA + EnbPI
        colors = cm.rainbow(np.linspace(0, 1, num_method))
        mtds = np.append('ARIMA', muh_fun)
        # label_names = methods_name
        label_names = {'ARIMA': 'ARIMA', 'RidgeCV': 'EnbPI Ridge',
                       'RandomForestRegressor': 'EnbPI RF', 'Sequential': 'EnbPI NN', 'RNN': 'EnbPI RNN'}
        # if 'ARIMA' in methods_name:
        #     colors = ['orange', 'red', 'blue', 'black']
        # else:
        #     colors = ['red', 'blue', 'black']
        first = 0
        second = 1
        if one_D:
            first = 2
            second = 3
        axisfont = 20
        titlefont = 24
        tickfont = 16
        name = 'mean'
        for i in range(num_method):
            if two_rows:
                # Coverage
                ax[j, first].plot(x_axis, cov_together[i][name], linestyle='-',
                                  marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, first].fill_between(x_axis, cov_together[i][name]-cov_together[i]['se'],
                                          cov_together[i][name]+cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, first].set_ylim(0, 1)
                ax[j, first].tick_params(axis='both', which='major', labelsize=tickfont)
                # Width
                ax[j, second].plot(x_axis, width_together[i][name], linestyle='-',
                                   marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, second].fill_between(x_axis, width_together[i][name]-width_together[i]['se'],
                                           width_together[i][name]+width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, second].tick_params(axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[j, first].plot(x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                nrow = len(Dataname)
                ax[nrow-1, 0].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[nrow-1, 1].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
            else:
                # Coverage
                ax[first].plot(x_axis, cov_together[i][name], linestyle='-',
                               marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[first].fill_between(x_axis, cov_together[i][name]-cov_together[i]['se'],
                                       cov_together[i][name]+cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[first].set_ylim(0, 1)
                ax[first].tick_params(axis='both', which='major', labelsize=tickfont)
                # Width
                ax[second].plot(x_axis, width_together[i][name], linestyle='-',
                                marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[second].fill_between(x_axis, width_together[i][name]-width_together[i]['se'],
                                        width_together[i][name]+width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[second].tick_params(axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[first].plot(x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                ax[first].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[second].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
        if two_rows:
            j += 1
        else:
            one_D = True
    if two_rows:
        ax[0, 0].set_title('Coverage', fontsize=axisfont)
        ax[0, 1].set_title('Width', fontsize=axisfont)
    else:
        ax[0].set_title('Coverage', fontsize=axisfont)
        ax[1].set_title('Width', fontsize=axisfont)
        ax[2].set_title('Coverage', fontsize=axisfont)
        ax[3].set_title('Width', fontsize=axisfont)
    if two_rows:
        ax[0, 0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[1, 0].set_ylabel('Unitivariate', fontsize=axisfont)
    else:
        ax[0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[2].set_ylabel('Unitivariate', fontsize=axisfont)
    fig.tight_layout(pad=0)
    if two_rows:
        # ax[0, 1].legend(loc='upper left', fontsize=axisfont-2)
        ax[1, 1].legend(loc='upper center',
                        bbox_to_anchor=(-0.08, -0.18), ncol=3, fontsize=axisfont-2)
    else:
        # ax[1].legend(loc='upper left', fontsize=axisfont-2)
        # ax[3].legend(loc='upper left', fontsize=axisfont-2)
        ax[3].legend(loc='upper center',
                     bbox_to_anchor=(-0.75, -0.18), ncol=5, fontsize=axisfont-2)
    if save:
        if two_rows:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
        else:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}_one_row.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)


def grouped_box_new(dataname, type, extra_save=''):
    '''First (Second) row contains grouped boxplots for multivariate (univariate) for Ridge, RF, and NN.
       Each boxplot contains coverage and width for all three PI methods over 3 (0.1, 0.3, 0.5) train/total data, so 3*3 boxes in total
       extra_save is for special suffix of plot (such as comparing NN and RNN)'''
    font_size = 18
    label_size = 20
    results = pd.read_csv(f'Results/{dataname}_many_train_new{extra_save}.csv')
    results.sort_values('method', inplace=True, ascending=True)
    results.loc[results.method == 'Ensemble', 'method'] = 'EnbPI'
    results.loc[results.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    results_1d = pd.read_csv(f'Results/{dataname}_many_train_new_1d{extra_save}.csv')
    results_1d.sort_values('method', inplace=True, ascending=True)
    results_1d.loc[results_1d.method == 'Ensemble', 'method'] = 'EnbPI'
    results_1d.loc[results_1d.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    if 'Sequential' in np.array(results.muh_fun):
        results['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
        results_1d['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
    regrs = np.unique(results.muh_fun)
    regrs_label = {'RidgeCV': 'Ridge', 'LassoCV': 'Lasso', 'RandomForestRegressor': "RF",
                   'NeuralNet': "NN", 'RNN': 'RNN', 'GaussianProcessRegressor': 'GP'}
    # Set up plot
    ncol = 2  # Compare RNN vs NN
    if len(regrs) > 2:
        ncol = 3  # Ridge, RF, NN
        regrs = ['RidgeCV', 'NeuralNet', 'RNN']
    if type == 'coverage':
        f, ax = plt.subplots(2, ncol, figsize=(3*ncol, 6), sharex=True, sharey=True)
    else:
        # all plots in same row share y-axis
        f, ax = plt.subplots(2, ncol, figsize=(3*ncol, 6), sharex=True, sharey=True)
    f.tight_layout(pad=0)
    # Prepare for plot
    d = 20
    results_1d.train_size += d  # for plotting purpose
    tot_data = int(max(results.train_size)/0.278)
    results['ratio'] = np.round(results.train_size/tot_data, 2)
    results_1d['ratio'] = np.round(results_1d.train_size/tot_data, 2)
    j = 0  # column, denote aggregator
    ratios = np.unique(results['ratio'])
    # train_size_for_plot = [ratios[2], ratios[4], ratios[6], ratios[9]] # This was for 4 boxplots in one figure
    train_size_for_plot = ratios
    for regr in regrs:
        mtd = ['EnbPI', 'ICP', 'Weighted ICP']
        mtd_colors = ['red', 'royalblue', 'black']
        color_dict = dict(zip(mtd, mtd_colors))  # specify colors for each box
        # Start plotting
        which_train_idx = [fraction in train_size_for_plot for fraction in results.ratio]
        which_train_idx_1d = [fraction in train_size_for_plot for fraction in results_1d.ratio]
        results_plt = results.iloc[which_train_idx, ]
        results_1d_plt = results_1d.iloc[which_train_idx_1d, ]
        sns.boxplot(y=type, x='ratio',
                              data=results_plt[results_plt.muh_fun == regr],
                              palette=color_dict,
                              hue='method', ax=ax[0, j], showfliers=False)
        sns.boxplot(y=type, x='ratio',
                    data=results_1d_plt[results_1d_plt.muh_fun == regr],
                    palette=color_dict,
                    hue='method', ax=ax[1, j], showfliers=False)
        for i in range(2):
            ax[i, j].tick_params(axis='both', which='major', labelsize=14)
            if type == 'coverage':
                ax[i, j].axhline(y=0.9, color='black', linestyle='dashed')
            # Control legend
            ax[i, j].get_legend().remove()
            # Control y and x-label
            if j == 0:
                # Y-label on
                ax[0, 0].set_ylabel('Multivariate', fontsize=label_size)
                ax[1, 0].set_ylabel('Univariate', fontsize=label_size)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            else:
                y_label = ax[i, j].axes.get_yaxis().get_label()
                y_label.set_visible(False)
                if type == 'coverage':
                    # Y-label off
                    y_axis = ax[i, j].axes.get_yaxis()
                    y_axis.set_visible(False)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            # Control Title
            if i == 0:
                ax[0, j].set_title(regrs_label[regr], fontsize=label_size)
        j += 1
        # Legend lastly
    # Assign to top middle
    # ax[1, 1].legend(loc='upper center',
    #                 bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=font_size)
    plt.legend(loc='upper center',
               bbox_to_anchor=(-0.15, -0.25), ncol=3, fontsize=font_size)
    plt.savefig(
        f'{dataname}_boxplot_{type}{extra_save}.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


def grouped_box_new_with_JaB(dataname):
    '''First (Second) row contains grouped boxplots for multivariate (univariate) for Ridge, RF, and NN.
       Each boxplot contains coverage and width for all three PI methods over 3 (0.1, 0.3, 0.5) train/total data, so 3*3 boxes in total
       extra_save is for special suffix of plot (such as comparing NN and RNN)'''
    font_size = 18
    label_size = 20
    results = pd.read_csv(f'Results/{dataname}_many_train_new_with_JaB.csv')
    results.sort_values('method', inplace=True, ascending=True)
    results.loc[results.method == 'Ensemble', 'method'] = 'EnbPI'
    # results.loc[results.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    results.loc[results.method == 'ICP',
                'method'] = 'Split Conformal, or Chernozhukov etal (2018,2020)'
    results.loc[results.method == 'JaB', 'method'] = 'J+aB (Kim etal 2020)'
    if 'Sequential' in np.array(results.muh_fun):
        results['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
    regrs = np.unique(results.muh_fun)
    regrs_label = {'RidgeCV': 'Ridge', 'LassoCV': 'Lasso', 'RandomForestRegressor': "RF",
                   'NeuralNet': "NN", 'RNN': 'RNN', 'GaussianProcessRegressor': 'GP'}
    # Set up plot
    ncol = 2  # Compare RNN vs NN
    if len(regrs) > 2:
        ncol = 6  # Ridge, RF, NN
        regrs = ['RidgeCV', 'NeuralNet', 'RNN']
    f, ax = plt.subplots(1, ncol, figsize=(3*ncol, 3), sharex=True)
    # f.tight_layout(pad=0)
    plt.tight_layout(pad=1.5)
    plt.set_share_axes(ax[:3], sharey=True)
    plt.set_share_axes(ax[3:], sharey=True)
    # Prepare for plot
    tot_data = int(max(results.train_size)/0.278)
    results['ratio'] = np.round(results.train_size/tot_data, 2)
    j = 0  # column, denote aggregator
    ratios = np.unique(results['ratio'])
    # train_size_for_plot = [ratios[2], ratios[4], ratios[6], ratios[9]] # This was for 4 boxplots in one figure
    train_size_for_plot = ratios
    for regr in regrs:
        # mtd = ['EnbPI', 'ICP', 'Weighted ICP']
        # mtd_colors = ['red', 'royalblue', 'black']
        mtd = [
            'EnbPI', 'Split Conformal, or Chernozhukov etal (2018,2020)', 'J+aB (Kim etal 2020)']
        mtd_colors = ['red', 'royalblue', 'black']
        color_dict = dict(zip(mtd, mtd_colors))  # specify colors for each box
        # Start plotting
        which_train_idx = [fraction in train_size_for_plot for fraction in results.ratio]
        results_plt = results.iloc[which_train_idx, ]
        ax1 = sns.boxplot(y='coverage', x='ratio',
                          data=results_plt[results_plt.muh_fun == regr],
                          palette=color_dict,
                          hue='method', ax=ax[j], showfliers=False, width=1, saturation=1)
        ax2 = sns.boxplot(y='width', x='ratio',
                          data=results_plt[results_plt.muh_fun == regr],
                          palette=color_dict,
                          hue='method', ax=ax[j+3], showfliers=False, width=1, saturation=1)
        for i, artist in enumerate(ax1.artists):
            if i % 3 == 0:
                col = mtd_colors[0]
            elif i % 3 == 1:
                col = mtd_colors[1]
            else:
                col = mtd_colors[2]
            # This sets the color for the main box
            artist.set_edgecolor(col)
        for i, artist in enumerate(ax2.artists):
            if i % 3 == 0:
                col = mtd_colors[0]
            elif i % 3 == 1:
                col = mtd_colors[1]
            else:
                col = mtd_colors[2]
            # This sets the color for the main box
            artist.set_edgecolor(col)
            # for k in range(6*i, 6*(i+1)):
            #     ax2.lines[k].set_color(col)
        ax[j].tick_params(axis='both', which='major', labelsize=14)
        if j <= 2:
            ax[j].axhline(y=0.9, color='black', linestyle='dashed')
        # Control legend
        ax[j].get_legend().remove()
        ax[j+3].get_legend().remove()
        # Control y and x-label
        ax[j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
        ax[j+3].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
        if j == 0:
            # Y-label on
            ax[j].set_ylabel('Coverage', fontsize=label_size)
            ax[j+3].set_ylabel('Width', fontsize=label_size)
        else:
            y_label = ax[j].axes.get_yaxis().get_label()
            y_label.set_visible(False)
            y_axis = ax[j].axes.get_yaxis()
            y_axis.set_visible(False)
            y_label = ax[j+3].axes.get_yaxis().get_label()
            y_label.set_visible(False)
            y_axis = ax[j+3].axes.get_yaxis()
            y_axis.set_visible(False)
        # Control Title
        ax[j].set_title(regrs_label[regr], fontsize=label_size)
        ax[j+3].set_title(regrs_label[regr], fontsize=label_size)
        j += 1
        # Legend lastly
    # plt.legend(loc='upper center',
    #            bbox_to_anchor=(-0.15, -0.25), ncol=3, fontsize=font_size)
    plt.legend(loc='upper center',
               bbox_to_anchor=(-2, -0.2), ncol=3, fontsize=font_size)
    plt.savefig(
        f'{dataname}_boxplot_rebuttal.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


'''For Conditional Coverage__Plotting'''


def PI_on_series_plus_cov_or_not(results, stride, which_hours, which_method, regr_method, Y_predict, no_slide=False, five_in_a_row=True):
    # Plot PIs on predictions for the particular hour
    # At most three plots in a row (so that figures look appropriately large)
    plt.rcParams.update({'font.size': 18})
    if five_in_a_row:
        ncol = 5
    else:
        ncol = 3
    nrow = np.ceil(len(which_hours)/ncol).astype(int)
    if stride == 24 or stride == 14 or stride == 15:
        # Multi-row
        fig, ax = plt.subplots(nrow*2, ncol, figsize=(ncol*4, nrow*5), sharex='row',
                               sharey='row', constrained_layout=True)
    else:
        fig, ax = plt.subplots(2, 5, figsize=(5*4, 5), sharex='row',
                               sharey='row', constrained_layout=True)
    if stride > 24:
        n1 = int(results[0].shape[0]/5)  # Because we focused on near-noon-data
    else:
        n1 = int(results[0].shape[0]/stride)
    plot_length = 91  # Plot 3 months, April-June
    method_ls = {'Ensemble': 0, 'ICP': 1, 'WeightedICP': 2}
    results_by_method = results[method_ls[which_method]]
    for i in range(len(which_hours)):
        hour = which_hours[i]
        if stride > 24:
            indices_at_hour = np.arange(n1)*5+hour
        else:
            indices_at_hour = np.arange(n1)*stride+hour
        to_plot = indices_at_hour[:plot_length]
        row = (i//ncol)*2
        col = np.mod(i, ncol)
        covered_or_not = []
        for j in range(n1):
            if Y_predict[indices_at_hour[j]] >= results_by_method['lower'][indices_at_hour[j]] and Y_predict[indices_at_hour[j]] <= results_by_method['upper'][indices_at_hour[j]]:
                covered_or_not.append(1)
            else:
                covered_or_not.append(0)
        coverage = np.mean(covered_or_not)
        coverage = np.round(coverage, 2)
        # Plot PI on data
        train_size = 92
        rot_angle = 15
        x_axis = np.arange(plot_length)
        if stride == 24 or stride == 14 or stride == 15:
            current_figure = ax[row, col]
        else:
            col = np.mod(i, 5)
            current_figure = ax[0, col]
        current_figure.scatter(x_axis, Y_predict[to_plot], marker='.', s=3, color='black')
        current_figure.plot(x_axis, np.maximum(0, results_by_method['upper'][to_plot]))
        current_figure.plot(x_axis, np.maximum(0, results_by_method['lower'][to_plot]))
        xticks = np.linspace(0, plot_length-30, 3).astype(int)  # For axis purpose, subtract June
        xtick_labels = [calendar.month_name[int(i/30)+4]
                        for i in xticks]  # Get months, start from April
        current_figure.set_xticks(xticks)
        current_figure.set_xticklabels(xtick_labels)
        current_figure.tick_params(axis='x', rotation=rot_angle)
        # Title
        if stride == 24:
            current_figure.set_title(f'At {hour}:00 \n Coverage is {coverage}')
        elif stride == 5 or no_slide:
            current_figure.set_title(f'At {hour+10}:00 \n Coverage is {coverage}')
        else:
            if stride == 15:
                current_figure.set_title(f'At {hour+5}:00 \n Coverage is {coverage}')
            else:
                current_figure.set_title(f'At {hour+6}:00 \n Coverage is {coverage}')
        # if stride == 14:
        #     # Sub data`
        #     current_figure.set_title(f'At {hour+6}:00 \n Coverage is {coverage}')
        # elif stride == 24:
        #     # Full data
        #     current_figure.set_title(f'At {hour}:00 \n Coverage is {coverage}')
        # else:
        #     # Near noon data
        #     current_figure.set_title(f'At {hour+10}:00 \n Coverage is {coverage}')
        # Plot cover or not over test period
        x_axis = np.arange(n1)
        if stride == 24 or stride == 14 or stride == 15:
            current_figure = ax[row+1, col]
        else:
            col = np.mod(i, 5)
            current_figure = ax[1, col]
        current_figure.scatter(x_axis, covered_or_not, marker='.', s=0.4)
        current_figure.set_ylim([-1, 2])
        xticks = np.linspace(0, n1-31, 3).astype(int)  # For axis purpose, subtract December
        xtick_labels = [calendar.month_name[int(i/30)+4] for i in xticks]  # Get months
        current_figure.set_xticks(xticks)
        current_figure.set_xticklabels(xtick_labels)
        current_figure.tick_params(axis='x', rotation=rot_angle)
        yticks = [0, 1]
        current_figure.set_yticks(yticks)
        current_figure.set_yticklabels(['Uncovered', 'Covered'])
        # xticks = current_figure.get_xticks()  # Actual numbers
        # xtick_labels = [f'T+{int(i)}' for i in xticks]
        # current_figure.set_xticklabels(xtick_labels)
    # if no_slide:
    #     fig.suptitle(
    #         f'EnbPI Intervals under {regr_method} without sliding', fontsize=22)
    # else:
    #     fig.suptitle(
    #         f'EnbPI Intervals under {regr_method} with s={stride}', fontsize=22)
    return fig


def make_cond_plots(Data_name, results_ls, no_slide, missing, one_d, five_in_a_row=True):
    for data_name in Data_name:
        #result_ridge, result_rf, result_nn, stride, Y_predict = results_ls[data_name]
        result_ridge, result_rf, stride, Y_predict = results_ls[data_name]
        #res = [result_ridge, result_rf, result_nn]
        res = [result_ridge, result_rf]
        if no_slide:
            which_hours = [0, 1, 2, 3, 4]  # 10AM-2PM
        else:
            if stride == 24:
                if five_in_a_row:
                    which_hours = [7, 8, 9, 16, 17, 10, 11, 12, 13, 14]
                else:
                    which_hours = [7, 8, 10, 11, 12, 13, 14, 16, 17]
            elif stride == 5:
                which_hours = [0, 1, 2, 3, 4]
            else:
                if five_in_a_row:
                    if data_name == 'Solar_Atl':
                        which_hours = [i-6 for i in [7, 8, 9, 16, 17, 10, 11, 12, 13, 14]]
                    else:
                        which_hours = [i-5 for i in [7, 8, 9, 16, 17, 10, 11, 12, 13, 14]]
                else:
                    if data_name == 'Solar_Atl':
                        # which_hours = [i-6 for i in [7, 8, 10, 11, 12, 13, 14, 16, 17]]
                        which_hours = [i-6 for i in [8, 9, 16, 11, 12, 13]]
                    else:
                        # which_hours = [i-5 for i in [7, 8, 10, 11, 12, 13, 14, 16, 17]]
                        which_hours = [i-5 for i in [8, 9, 16, 11, 12, 13]]
        which_method = 'Ensemble'
        regr_methods = {0: 'Ridge', 1: 'RF', 2: 'NN'}
        X_data_type = {True: 'uni', False: 'multi'}
        Xtype = X_data_type[one_d]
        slide = '_no_slide' if no_slide else '_daily_slide'
        Dtype = {24: '_fulldata', 14: '_subdata', 15: '_subdata', 5: '_near_noon_data'}
        if no_slide:
            dtype = ''
        else:
            dtype = Dtype[stride]
        miss = '_with_missing' if missing else ''
        for i in range(len(res)):
            regr_method = regr_methods[i]
            fig = PI_on_series_plus_cov_or_not(
                res[i], stride, which_hours, which_method, regr_method, Y_predict, no_slide, five_in_a_row)
            fig.savefig(f'{data_name}_{regr_method}_{Xtype}_PI_on_series_plus_cov_or_not{slide}{dtype}{miss}.pdf', dpi=300, bbox_inches='tight',
                        pad_inches=0)


########################################################################################################################
########################################################################################################################
##############################################    PI_class_EnbPI.py    #################################################
########################################################################################################################
########################################################################################################################
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
                callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
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
