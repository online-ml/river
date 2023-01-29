# Conformal prediction implementation in River

- BOGGIO Richard
- MDIHI Samy
- VERON Marc

This Conformal Prediction Implementation relies on the paper by Margaux Zaffran et al. "Adaptative Conformal Predictions for Time series" (https://arxiv.org/abs/2202.07282). This paper has 2 parts: expert aggregation for regression or classification, and the definition of **confidence intervals on streaming data**. We focus here on the implementation in River of these confidence interval estimation techniques. We rely on the work of the research group, implemented in Python and available on github : https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries

Conformal prediction is a general term for identifying confidence interval definition methods in machine learning that go beyond the simple gaussian approach. 

The aim of our work is to allow users to benefit from these method when using regression and prediction models with River. So we increased the **conf** module first. This one is present on the River git repo, but is not deployed on the downloadable Python version. It contains the parent class **interval** that is used as base for the different methods: **Gaussian**, **ConformalPrediction**, **AdaptativeConformalPrediction**. Next, we augment the **time_series** module, in which we update the evaluation method to allow for intervals at different horizons. Indeed the logic of this module is to predict not only at horizon 1, but further. The calculation of intervals must therefore be integrated into this way, hence the base definition in conf. 

To ensure the integration of all these methods, the **\_\_init\_\_** and **base** files have been updated. This allows to have an almost functional environment as described below. A notebook is also provided in our Git repo : https://github.com/mverontarabeux/river/tree/ConformalPrediction


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install river.

```bash
pip install river
```

## Usage

```python
# import the relevant River modules
from river import datasets, metrics
from time_series.holt_winters import HoltWinters

# import the interval methods from the custom conf module (defined as a folder)
import time_series
import conf.ACP
import conf.CP
import conf.gaussian

# Get some data
dataset = datasets.AirlinePassengers()

# Define a forecasting model 
model = HoltWinters(
        alpha=0.3,
        beta=0.1,
        gamma=0.6,
        seasonality=12,
        multiplicative=True
        )

# Set the metric and interval methods
metric = metrics.MAE()
interval = conf.gaussian.Gaussian(window_size=calib_period, alpha=0.10)

# Evaluate the model
time_series.evaluate(dataset,
                     model,
                     metric,
                     interval,
                     horizon=12,
                     residual_calibration_period = calib_period
                     )

```
## Contribution

A pull requests has been sent with the latest updates. 
The code is flake8 compliant.

Please also check the Notebook_ConformalPrediction.ipynb for more insight.