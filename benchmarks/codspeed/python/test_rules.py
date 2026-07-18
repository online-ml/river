from marks import benchmark
from workloads import regression_stream

from river import rules


@benchmark("rules")
def test_amrules_learn(benchmark) -> None:
    stream = regression_stream(500)

    def run() -> None:
        model = rules.AMRules()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
