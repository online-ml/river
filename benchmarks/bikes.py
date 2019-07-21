from creme import compat
from creme import compose
from creme import datasets
from creme import feature_extraction
from creme import linear_model
from creme import metrics
from creme import preprocessing
from creme import optim
from creme import stats
from sklearn import linear_model as sk_linear_model
import torch

import benchmark
import wrap


def main():

    lr = 0.005

    torch_model = torch.nn.Linear(in_features=6, out_features=1, bias=True)
    torch.nn.init.constant_(torch_model.weight, 0)
    torch.nn.init.constant_(torch_model.bias, 0)

    def add_hour(x):
        x['hour'] = x['moment'].hour
        return x

    benchmark.benchmark(
        get_X_y=datasets.fetch_bikes,
        n=182470,
        get_pp=lambda: (
            compose.Whitelister('clouds', 'humidity', 'pressure', 'temperature', 'wind') +
            (
                add_hour |
                feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
            ) |
            preprocessing.StandardScaler()
        ),
        models=[
            ('PyTorch (CPU)', 'Linear', wrap.PyTorchRegressor(
                model=torch_model,
                loss_fn=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(torch_model.parameters(), lr=lr)
            )),

            ('Vowpal Wabbit', 'Linear', wrap.VowpalWabbitRegressor(
                loss_function='squared',
                sgd=True,
                learning_rate=lr
            )),


            ('creme', 'LinearRegression', linear_model.LinearRegression(
                optimizer=optim.VanillaSGD(lr),
                l2=0.,
                intercept_lr=lr
            )),

            ('scikit-learn', 'SGD', wrap.ScikitLearnRegressor(
                sklearn_estimator=sk_linear_model.SGDRegressor(
                    learning_rate='constant',
                    eta0=lr,
                    penalty='none'
                ),
            ))
        ],
        get_metric=metrics.MSE
    )


if __name__ == '__main__':
    main()
