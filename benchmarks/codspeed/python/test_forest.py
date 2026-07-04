from marks import heavy
from workloads import N_PREDICT, binary_stream, regression_stream

from river import forest


@heavy("forest")
def test_arf_classifier_learn(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> None:
        model = forest.ARFClassifier(n_models=3, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@heavy("forest")
def test_arf_classifier_predict(benchmark) -> None:
    stream = binary_stream(500)
    model = forest.ARFClassifier(n_models=3, seed=42)
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_one(x)

    benchmark(run)


@heavy("forest")
def test_amf_classifier_learn(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> None:
        model = forest.AMFClassifier(n_estimators=3, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@heavy("forest")
def test_amf_classifier_predict(benchmark) -> None:
    stream = binary_stream(500)
    model = forest.AMFClassifier(n_estimators=3, seed=42)
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_one(x)

    benchmark(run)


@heavy("forest")
def test_arf_regressor_learn(benchmark) -> None:
    stream = regression_stream(500)

    def run() -> None:
        model = forest.ARFRegressor(n_models=3, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
