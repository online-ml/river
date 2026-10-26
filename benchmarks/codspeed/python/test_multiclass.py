from marks import benchmark
from workloads import multiclass_stream

from river import linear_model, multiclass


@benchmark("multiclass")
def test_one_vs_rest_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
