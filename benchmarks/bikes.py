from creme import compat
from creme import compose
from creme import datasets
from creme import feature_extraction
from creme import linear_model
from creme import meta
from creme import metrics
from creme import preprocessing
from creme import optim
from creme import stats
from sklearn import linear_model as sk_linear_model

import benchmark


def main():

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
            # ('creme', 'LinReg', linear_model.LinearRegression(
            #     optimizer=optim.VanillaSGD(0.01),
            #     l2=0.
            # )),

            ('creme', 'GLM', linear_model.GLMRegressor(
                optimizer=optim.VanillaSGD(0.01),
                l2=0.
            )),

            ('creme', 'GLM', meta.Detrender(linear_model.GLMRegressor(
                optimizer=optim.VanillaSGD(0.01),
                l2=0.
            ))),



            # ('sklearn', 'SGD', compat.CremeRegressorWrapper(
            #     sklearn_estimator=sk_linear_model.SGDRegressor(
            #         learning_rate='constant',
            #         eta0=0.01,
            #         fit_intercept=True,
            #         penalty='none'
            #     ),
            # )),
            # ('sklearn', 'SGD no intercept', compat.CremeRegressorWrapper(
            #     sklearn_estimator=sk_linear_model.SGDRegressor(
            #         learning_rate='constant',
            #         eta0=0.01,
            #         fit_intercept=False,
            #         penalty='none'
            #     ),
            # )),
        ],
        get_metric=metrics.MSE
    )


if __name__ == '__main__':
    main()
