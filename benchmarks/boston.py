import functools

from creme import compat
from creme import linear_model
from creme import metrics
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
from sklearn import linear_model as sk_linear_model

import benchmark


def main():

    benchmark.benchmark(
        output_name='boston',
        get_X_y=functools.partial(stream.iter_sklearn_dataset, datasets.load_boston()),
        n=506,
        get_pp=preprocessing.StandardScaler,
        models=[
            ('creme', 'LinReg', linear_model.LinearRegression(
                optimizer=optim.VanillaSGD(0.001),
                l2=0.,
                intercept_lr=0.001
            )),
            ('sklearn', 'SGD', compat.CremeRegressorWrapper(
                sklearn_estimator=sk_linear_model.SGDRegressor(
                    learning_rate='constant',
                    eta0=0.001,
                    fit_intercept=True,
                    penalty='none'
                ),
            )),
        ],
        get_metric=metrics.MSE
    )


if __name__ == '__main__':
    main()
