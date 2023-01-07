import pickle

def read_pickle(path):
    with open(path,'rb') as f:
        file = pickle.load(f)
    return file

def write_pickle(file, path):
    with open(path,'wb') as f:
        pickle.dump(file, f)

def get_name_data(n, regression='Friedman', noise='ARMA', params_reg={}, params_noise={}, seed=1):
    """ ...

    Parameters
    ----------

    n : experiment sample size
    regression : regression model, can be Friedman
    noise : noise type, can be ARMA
    params_reg : parameters of the regression part
    params_noise : parameters of the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility used in the experiment

    Returns
    -------

    name :

    """

    assert regression in ['Friedman','Linear'], 'regression must be Friedman or Linear.'
    if regression == 'Friedman':
        name = 'Friedman_'
    elif regression == 'Linear':
        name = 'Linear_'

    assert noise in ['ARMA'], 'noise must be ARMA.'
    if noise == 'ARMA':
        ar = params_noise['ar']
        ma = params_noise['ma']

        ar_name = 'AR'
        for p in range(1,len(ar)):
            ar_name = ar_name + '_' + str(-ar[p])

        ma_name = 'MA'
        for q in range(1,len(ma)):
            ma_name = ma_name + '_' + str(ma[q])

        name = name + 'ARMA_' + ar_name + '_' + ma_name

        if 'scale' in params_noise:
            name = name + '_scale_' + str(params_noise['scale'])

        if 'process_variance' in params_noise:
            name = name + '_fixed_variance_' + str(int(params_noise['process_variance']))

    name = name + '_seed_' + str(int(seed)) + '_n_' + str(int(n))

    return name

def get_name_results(method, n=None, online=True, randomized=False, params_method={}, basemodel='RF', regression=None, noise=None, params_reg={}, params_noise={}, dataset=None):
    """ ...

    Parameters
    ----------

    method :
    params_method :

    Returns
    -------

    name :

    """

    # Results file name, depending on the method

    name_method = method + '_' + basemodel

    if (method == 'ACP') & (params_method != {}):
        name_method = name_method + '_gamma_' + str(params_method['gamma'])

    if randomized:
        name_method = name_method+'_randomized'

    if not online:
        name_method = name_method+'_offline'

    # Results directory name, depending on the data simulation

    #assert regression in ['Friedman','Linear'], 'regression must be Friedman or Linear.'
    if regression == 'Friedman':
        name_directory = 'Friedman_'
    elif regression == 'Linear':
        name_directory = 'Linear_'

    #assert noise in ['ARMA'], 'noise must be ARMA.'
    if noise == 'ARMA':
        ar = params_noise['ar']
        ma = params_noise['ma']

        ar_name = 'AR'
        for p in range(1,len(ar)):
            ar_name = ar_name + '_' + str(-ar[p])

        ma_name = 'MA'
        for q in range(1,len(ma)):
            ma_name = ma_name + '_' + str(ma[q])

        name_directory = name_directory + 'ARMA_' + ar_name + '_' + ma_name

        if 'scale' in params_noise:
            name_directory = name_directory + '_scale_' + str(params_noise['scale'])

        if 'process_variance' in params_noise:
            name_directory = name_directory + '_fixed_variance_' + str(int(params_noise['process_variance']))

    if dataset is not None:
        name_directory = dataset
    else:
        name_directory = name_directory + '_n_' + str(int(n))

    return name_directory, name_method
