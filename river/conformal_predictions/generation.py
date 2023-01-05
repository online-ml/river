import random
import numpy as np
import statsmodels as sm
import statsmodels.tsa.arima_process

def generate_data(n, regression='Friedman', noise='ARMA', params_reg={}, params_noise={}, seed=1):
    """

    Parameters
    ----------

    n : sample size to generate
    regression : regression model, can be Friedman
    noise : noise type, can be ARMA
    params_reg : parameters for the regression part
    params_noise : parameters for the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility

    Returns
    -------

    X : covariates values, array of size dxn
    Y : response values, array of size n

    """

    random.seed(seed)
    np.random.seed(seed)

    assert regression in ['Friedman', 'Linear'], 'regression must be Friedman or Linear.'
    if regression == 'Friedman':
        d = 6
        X = np.random.uniform(low=0,high=1,size=(d,n))
        Y_reg = 10*np.sin(np.pi*X[0]*X[1])+20*(X[2]-0.5)**2+10*X[3]+5*X[4]
    elif regression == 'Linear':
        d = 6
        X = np.random.uniform(low=0,high=1,size=(d,n))
        beta = np.random.uniform(low=-5,high=5,size=(d,1))
        Y_reg = X.T.dot(beta).reshape(n,)

    assert noise in ['ARMA'], 'noise must be ARMA.'
    if noise == 'ARMA':
        ar = params_noise['ar']
        ma = params_noise['ma']
        if 'scale' in params_noise:
            eps = sm.tsa.arima_process.arma_generate_sample(ar, ma, n, scale=params_noise['scale'])
        elif 'process_variance' in params_noise:
            v = params_noise['process_variance']
            p = len(ar)
            q = len(ma)
            if (p==2) and (q==1):
                # then it is an AR(1)
                scale = np.sqrt(v*(1-ar[1]**2))
            elif (p==1) and (q==2):
                # then it is an MA(1)
                scale = np.sqrt(v/(1+ma[1]**2))
            elif (p==2) and (q == 2):
                # then it is an ARMA(1,1)
                scale = np.sqrt(v*(1-ar[1]**2)/(1-2*ar[1]*ma[1]+ma[1]**2))
            elif (p==1) and (q == 1):
                # just a WN
                scale = np.sqrt(v)
            eps = sm.tsa.arima_process.arma_generate_sample(ar, ma, n, scale=scale)
        else:
            eps = sm.tsa.arima_process.arma_generate_sample(ar, ma, n)

    Y = Y_reg + eps

    return X, Y

def generate_multiple_data(n, regression='Friedman', noise='ARMA', params_reg={}, params_noise={}, seed_max=1):
    """ 

    Parameters
    ----------

    n : sample size to generate
    regression : regression model, can be Friedman
    noise : noise type, can be ARMA
    params_reg : parameters for the regression part
    params_noise : parameters for the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed_max : random seeds for reproducibility, will generate seed_max data-sets, of seeds 0 to seed_max-1

    Returns
    -------

    X : covariates values, array of size dxn
    Y : response values, array of size n

    """

    Xinit, _ = generate_data(n, regression=regression, noise=noise, params_reg=params_reg, params_noise=params_noise, seed=0)
    d = Xinit.shape[0]

    X = np.empty((seed_max,d,n))
    Y = np.empty((seed_max,n))

    for k in range(seed_max):
        Xk, Yk = generate_data(n, regression=regression, noise=noise, params_reg=params_reg, params_noise=params_noise, seed=k)
        X[k,:,:] = Xk
        Y[k,:] = Yk

    return X, Y
