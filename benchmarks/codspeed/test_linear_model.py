from marks import benchmark
from workloads import (
    N_PREDICT,
    binary_stream,
    high_dim_stream,
    multiclass_stream,
    regression_stream,
)

from river import linear_model, optim, preprocessing


@benchmark("linear_model")
def test_logistic_regression_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.005)
        )
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("linear_model")
def test_logistic_regression_predict(benchmark) -> None:
    stream = binary_stream()
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.005)
    )
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_proba_one(x)

    benchmark(run)


@benchmark("linear_model")
def test_linear_regression_learn(benchmark) -> None:
    stream = regression_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("linear_model")
def test_alma_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.ALMAClassifier()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("linear_model")
def test_pa_regressor_learn(benchmark) -> None:
    stream = regression_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.PARegressor(mode=2)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("linear_model")
def test_softmax_regression_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.SoftmaxRegression()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("linear_model")
def test_logistic_regression_high_dim_learn(benchmark) -> None:
    stream = high_dim_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
