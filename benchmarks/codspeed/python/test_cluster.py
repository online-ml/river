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
def test_dbstream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.DBSTREAM()
        for x in stream:
            model.learn_one(x)

    benchmark(run)
