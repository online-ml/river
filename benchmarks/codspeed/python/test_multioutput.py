from marks import benchmark
from workloads import multioutput_stream

from river import linear_model, multioutput


@benchmark("multioutput")
def test_classifier_chain_learn(benchmark) -> None:
    stream = multioutput_stream()

    def run() -> None:
        model = multioutput.ClassifierChain(linear_model.LogisticRegression(), order=[0, 1, 2])
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
