from marks import heavy
from workloads import high_dim_stream, text_stream

from river import feature_extraction


@heavy("feature_extraction")
def test_bag_of_words_transform(benchmark) -> None:
    stream = text_stream()
    model = feature_extraction.BagOfWords()

    def run() -> None:
        for text in stream:
            model.transform_one(text)

    benchmark(run)


@heavy("feature_extraction")
def test_tfidf_transform(benchmark) -> None:
    stream = text_stream()

    def run() -> None:
        model = feature_extraction.TFIDF()
        for text in stream:
            model.learn_one(text)
            model.transform_one(text)

    benchmark(run)


@heavy("feature_extraction")
def test_polynomial_extender_transform(benchmark) -> None:
    # Degree-2 expansion of 50 features yields 1,275 terms per sample: 200
    # samples are plenty of signal while keeping the CPU-simulation run short.
    stream = [x for x, _ in high_dim_stream(200)]
    model = feature_extraction.PolynomialExtender(degree=2)

    def run() -> None:
        for x in stream:
            model.transform_one(x)

    benchmark(run)
