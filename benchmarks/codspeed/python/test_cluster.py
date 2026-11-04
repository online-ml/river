from marks import benchmark
from workloads import binary_stream

from river import cluster


@benchmark("cluster")
def test_kmeans_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.KMeans(n_clusters=5, seed=42)
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_kmeans_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.KMeans(n_clusters=5, seed=42)
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_kmeans_predict_wide(benchmark) -> None:
    model = cluster.KMeans(n_clusters=5, seed=42)
    x = {i: (i % 17) / 17 for i in range(128)}
    model.predict_one(x)

    benchmark(lambda: model.predict_one(x))


@benchmark("cluster")
def test_dbstream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.DBSTREAM()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_textclust_learn(benchmark) -> None:
    stream = [
        {
            f"topic_{i % 32}": 2.0,
            f"shared_{i % 8}": 1.0,
            f"term_{i % 32}_{i % 12}": 1.0,
        }
        for i in range(1_000)
    ]

    def run() -> None:
        model = cluster.TextClust(
            radius=0.35,
            fading_factor=0.0005,
            tgap=100,
            real_time_fading=False,
        )
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_textclust_macro_cluster(benchmark) -> None:
    stream = [{f"term_{i}": 1.0, f"group_{i % 8}": 0.25} for i in range(128)]

    def run() -> None:
        model = cluster.TextClust(
            radius=0.01,
            fading_factor=0.00001,
            tgap=10_000,
            real_time_fading=False,
            auto_merge=False,
            num_macro=8,
        )
        for x in stream:
            model.learn_one(x)
        model.get_macroclusters()

    benchmark(run)
