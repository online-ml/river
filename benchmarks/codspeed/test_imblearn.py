from marks import benchmark
from workloads import binary_stream

from river import imblearn, linear_model


@benchmark("imblearn")
def test_random_under_sampler_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = imblearn.RandomUnderSampler(
            linear_model.LogisticRegression(),
            desired_dist={True: 0.5, False: 0.5},
            seed=42,
        )
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
