from marks import benchmark, heavy
from workloads import categorical_stream, fuzzy_category_stream, high_dim_stream

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


@heavy("preprocessing")
def test_gap_encoder_learn(benchmark) -> None:
    # Gamma-Poisson updates over a growing character n-gram vocabulary: the
    # heaviest path in GapEncoder. 300 fuzzy strings keep the CPU-simulation
    # run short while still exercising vocabulary growth and the E-step.
    stream = fuzzy_category_stream()

    def run() -> None:
        model = preprocessing.GapEncoder(n_components=10, seed=42)
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@heavy("preprocessing")
def test_gap_encoder_transform(benchmark) -> None:
    # transform_one is read-only; fit once outside the measured callable so the
    # benchmark isolates the E-step inference cost on a fixed model.
    stream = fuzzy_category_stream()
    model = preprocessing.GapEncoder(n_components=10, seed=42)
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.transform_one(x)

    benchmark(run)
