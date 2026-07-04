from marks import benchmark
from workloads import N_PREDICT, multiclass_stream, regression_stream

from river import tree


@benchmark("tree")
def test_hoeffding_tree_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = tree.HoeffdingTreeClassifier(grace_period=50)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("tree")
def test_hoeffding_tree_predict(benchmark) -> None:
    stream = multiclass_stream()
    model = tree.HoeffdingTreeClassifier(grace_period=50)
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_one(x)

    benchmark(run)


@benchmark("tree")
def test_hoeffding_adaptive_tree_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = tree.HoeffdingAdaptiveTreeClassifier(seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("tree")
def test_hoeffding_tree_regressor_learn(benchmark) -> None:
    stream = regression_stream()

    def run() -> None:
        model = tree.HoeffdingTreeRegressor(grace_period=50)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
