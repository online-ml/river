from marks import benchmark
from workloads import binary_stream

from river import calibration, linear_model


@benchmark("calibration")
def test_calibrated_classifier_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = calibration.CalibratedClassifier(linear_model.LogisticRegression(), lr=0.1)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)