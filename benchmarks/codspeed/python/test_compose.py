from marks import benchmark
from workloads import N_PREDICT, binary_stream

from river import feature_extraction, linear_model, preprocessing


@benchmark("compose")
def test_pipeline_predict_overhead(benchmark) -> None:
    stream = binary_stream()
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_proba_one(x)

    benchmark(run)


@benchmark("compose")
def test_transformer_union_transform(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = preprocessing.StandardScaler() + feature_extraction.PolynomialExtender(degree=2)

    def run() -> None:
        for x in stream:
            model.transform_one(x)

    benchmark(run)
