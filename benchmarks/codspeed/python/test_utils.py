from marks import benchmark
from workloads import binary_stream, high_dim_stream

from river import linear_model, utils


@benchmark("utils")
def test_vectordict_arithmetic(benchmark) -> None:
    dict_a = high_dim_stream(2)[0][0]
    dict_b = high_dim_stream(2)[1][0]
    u = utils.VectorDict(dict_a)
    v = utils.VectorDict(dict_b)

    def run() -> None:
        for _ in range(1_000):
            (u + v) * 2 - u
            u @ v

    benchmark(run)


@benchmark("utils")
def test_calibrated_classifier_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = utils.CalibratedClassifier(linear_model.LogisticRegression(), lr=0.1)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
