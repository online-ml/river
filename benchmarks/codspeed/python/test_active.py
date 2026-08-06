from marks import benchmark
from workloads import multiclass_stream

from river import active, naive_bayes


@benchmark("active")
def test_entropy_sampler_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = active.EntropySampler(naive_bayes.GaussianNB(), seed=42)
        for x, y in stream:
            _, ask = model.predict_one(x)
            if ask:
                model.learn_one(x, y)

    benchmark(run)
