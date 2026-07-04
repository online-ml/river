from marks import benchmark
from workloads import categorical_stream, high_dim_stream

from river import preprocessing


@benchmark("preprocessing")
def test_standard_scaler_learn_transform(benchmark) -> None:
    stream = [x for x, _ in high_dim_stream()]

    def run() -> None:
        model = preprocessing.StandardScaler()
        for x in stream:
            model.learn_one(x)
            model.transform_one(x)

    benchmark(run)


@benchmark("preprocessing")
def test_one_hot_encoder_transform(benchmark) -> None:
    stream = categorical_stream()

    def run() -> None:
        model = preprocessing.OneHotEncoder()
        for x in stream:
            model.learn_one(x)
            model.transform_one(x)

    benchmark(run)


@benchmark("preprocessing")
def test_feature_hasher_transform(benchmark) -> None:
    stream = categorical_stream()
    model = preprocessing.FeatureHasher(n_features=2**16, seed=42)

    def run() -> None:
        for x in stream:
            model.transform_one(x)

    benchmark(run)
