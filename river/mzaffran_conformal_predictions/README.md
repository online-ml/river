# Adaptive Conformal Predictions for Time Series

This directory contains implementations of the methods described in ["Adaptive Conformal Predictions for Time Series"](https://arxiv.org/abs/2202.07282), as well as details to reproduce the main figures of the paper.
The following notes provide help to use this code to benchmark new methods for CP in time series.

## Table of Contents

- [Organization](#organization)
- [Usage](#usage)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Planned improvements](#planned-improvements-of-this-repository)
- [License](#license)

## Organization

- `AgACI/` contains the R code og AgACI algorithm.
- `data/`contains all data pickle files generated.
- `data_prices/` contains the French electricity spot prices data set, built using data from **eco2mix**.
- `enbpi/` contains the code of the EnbPI algorithm, from Xu and Xie, ICML, 2021 (https://github.com/hamrel-cxu/EnbPI/tree/main), as well as a modification for EnbPI V2.
- `plots/` contains the plots produced by the jupyter notebooks (currently empty for size purposes).
- `results/` contains all results pickle files obtained.

## Usage

#### File ``main.py``

Allows to directly simulate data according to the data generation procedure described in Section 5.1, and to apply on it the wide range of methods considered (OSSCP, Offline SSCP, EnbPI, EnbPI V2, ACI with one gamma chosen). Further parameters can be given, that we recall in the following command line:

```shell
$ python main.py --alpha 0.1 --nrep 100 --ar -0.9 --process_variance 10 --methods "CP" "ACP" "EnbPI" --gamma 0.05
```

###### General parameters
- *nrep*: is the number of repetitions of the experiment, set to 500 in our study.
- *n*: equals to T_0+T_1 (default 300)
- *train*: is T_0 (default 200)

###### Noise parameters
- *ar*: is the list of ar parameters, that is phi in our setting, but one can give *p* values consecutively to produce an AR(p) (but then process_variance shall remain unfixed). Note that if you want phi = 0.9 (defined in our paper) you need to give as argument -0.9.
- *ma*: same than for *ar* but for theta values. Here, no need to take the opposite.
- *process_variance*: the value of the desired asymptotic noise process
- *scale*: value of the innovation (xi_t) standard deviation (can not be used with *process_variance*)

###### Methods parameters
- *methods*: the method considered among "ACP", "OSSCP", "SSCP", "EnbPI". Note that to use ACI you need to type "ACP". When setting EnbPI, by default EnbPI V2 is performed. To obtain original EnbPI, add --mean 1.
- *gamma*: fo "ACP" method to define completely ACI (default is 0.01)
- *B*: the number of bootstrap for EnbPI (default is 30)
- *mean*: if 1, then EnbPI V2 is performed instead of EnbPI (default 1)
- *online*: if 1, performs Online for SSCP, if 0 then performs Offline SSCP (default 1)
- *randomized*: if 1 splits randomly the data, if 0 the split is sequential (default 0)
- *cores*: the number of cores to used to parallelize the RF (default 1)

###### Overall command line

The provided command line will simulate 100 AR(1) with phi = 0.9 of length 300 and asymptotic variance 10, and will apply OSSCP, ACI with gamma = 0.01, and EnbPI V2, using 200 points for training.

The results are stored in the folder ``results``. This folder will contain one subfolder, named after Friedman, 300 and 0.9 which in turns will contain the files of the results for each method.

#### File ``main_acp.py``
Works similarly but the aim is here to apply different versions of ACI. To do so, the command line will contain the same argument as for main.py (except --mean for example, as it is not an applicable option for ACI). The list of gammas to be applied should be modified beforehands in the ``main_acp.py`` line 73.

#### Folder ``AgACI``
In the folder AgACI, a R project allows to wrap around the results produced by ``main_acp.py`` once they have been created. The list of gamma should also be specified.
This part is in R language, as the OPERA package is not yet available in python. OPERA allows to compute online agregation with expert advice for several rules, such as EWA, ML-poly or BOA.

## Reproducing the experiments

We explain how to obtain the Figures from the paper, using the previously described files. For repository size concerns, data and results of the different models are available only for the figures of the main part of the paper. We explain here how to re-generate these data and results, and the appendices plots can be obtained by adapting these explanations.

##### Figure 2

To obtain Figure 2, you need to run:

```shell
$ python ar_numerical.py --seed k
```
In our experiments we repeated this for seeds k from 0 to 24.
Then, to obtain the plot, just execute the jupyter notebook ``arma_numerical_plot.ipynb``.

Values of phi for the experiments should be changed in ``ar_numerical.py`` line 29 if you wish to.

##### Figure 3

```shell
$ python main_acp.py --alpha 0.1 --nrep 500 --ar -0.99 --ma 0.99 --process_variance 10
$ python main_acp.py --alpha 0.1 --nrep 500 --ar -0.95 --ma 0.95 --process_variance 10
$ python main_acp.py --alpha 0.1 --nrep 500 --ar -0.9 --ma 0.9 --process_variance 10
$ python main_acp.py --alpha 0.1 --nrep 500 --ar -0.8 --ma 0.9 --process_variance 10
$ python main_acp.py --alpha 0.1 --nrep 500 --ar -0.1 --ma 0.1 --process_variance 10
```

Then you can use the first part in ``AgACI/Script/acp_gamma.R``, until line 88.

Finally, use the jupyter notebook ``plots_gamma.ipynb``.

##### Figure 5

First do what is needed for Figure 3 so that you obtain AgACI. Then:

```shell
$ python main.py --alpha 0.1 --nrep 500 --ar -0.99 --ma 0.99 --process_variance 10 --methods "CP" "ACP" "EnbPI"
$ python main.py --alpha 0.1 --nrep 500 --ar -0.95 --ma 0.95 --process_variance 10 --methods "CP" "ACP" "EnbPI"
$ python main.py --alpha 0.1 --nrep 500 --ar -0.9 --ma 0.9 --process_variance 10 --methods "CP" "ACP" "EnbPI"
$ python main.py --alpha 0.1 --nrep 500 --ar -0.8 --ma 0.8 --process_variance 10 --methods "CP" "ACP" "EnbPI"
$ python main.py --alpha 0.1 --nrep 500 --ar -0.1 --ma 0.1 --process_variance 10 --methods "CP" "ACP" "EnbPI"
```

```shell
$ python main.py --alpha 0.1 --nrep 500 --ar -0.99 --ma 0.99 --process_variance 10 --methods="ACP" --gamma 0.05
$ python main.py --alpha 0.1 --nrep 500 --ar -0.95 --ma 0.95 --process_variance 10 --methods="ACP" --gamma 0.05
$ python main.py --alpha 0.1 --nrep 500 --ar -0.9 --ma 0.9 --process_variance 10 --methods="ACP" --gamma 0.05
$ python main.py --alpha 0.1 --nrep 500 --ar -0.8 --ma 0.8 --process_variance 10 --methods="ACP" --gamma 0.05
$ python main.py --alpha 0.1 --nrep 500 --ar -0.1 --ma 0.1 --process_variance 10 --methods="ACP" --gamma 0.05
```

```shell
$ python main.py --alpha 0.1 --nrep 500 --ar -0.99 --ma 0.99 --process_variance 10 --methods "CP" --online 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.95 --ma 0.95 --process_variance 10 --methods "CP" --online 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.9 --ma 0.9 --process_variance 10 --methods "CP" --online 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.8 --ma 0.8 --process_variance 10 --methods "CP" --online 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.1 --ma 0.1 --process_variance 10 --methods "CP" --online 0
```

```shell
$ python main.py --alpha 0.1 --nrep 500 --ar -0.99 --ma 0.99 --process_variance 10 --methods "EnbPI" --mean 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.95 --ma 0.95 --process_variance 10 --methods "EnbPI" --mean 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.9 --ma 0.9 --process_variance 10 --methods "EnbPI" --mean 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.8 --ma 0.8 --process_variance 10 --methods "EnbPI" --mean 0
$ python main.py --alpha 0.1 --nrep 500 --ar -0.1 --ma 0.1 --process_variance 10 --methods "EnbPI" --mean 0
```

Then you can execute the jupyter notebook ``plots.ipnyb`` until cell 19 to obtain Figure 5.
To execute the next cells, corresponding to figures in appendices, you need to generate the results by adapting the necessary command lines above and running them before going onto the notebook.

#### Application to French electricity spot prices forecasting

The folder data_prices contains ``Prices_2016_2019_extract.csv``, that is the considered French electricity spot prices data set from 2016 to 2019  with the explanatory variables used (coming from **eco2mix**). To apply the different methods, a notebook, ``Application_Spot_France.ipynb``, is provided containing all the codes, the cells should only be re-executed. This will especially give you Figures 6 and 7. Note that to add AgACI you need to run the cell in the notebooks for the different gammas (especially cell 12), then use the same file in ``AgACI/Scripts/acp_gamma.R`` until line 20 and then everything after line 113. Then go back to the notebook!

## Planned improvements of this repository

1. Update ACP name to be consistent with ACI (Gibbs & Candès)
2. Provide a Python code for AgACI, based on calling R from Python, surely using ``rpy2.py``.
3. Merge the functions to run with different gamma (``models.run_multiple_gamma_ACP``) and the baselines one (``models.run_experiments``)
4. Generalize all the repository so that any (?) regressor with a .fit and .predict method can be used (at least for ACI, SCP and Gaussian)

## License

[MIT](LICENSE) © Margaux Zaffran
