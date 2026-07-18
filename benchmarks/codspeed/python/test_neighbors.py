from marks import benchmark
from workloads import binary_stream

from river import neighbors

# Keep 500 samples: SWINN only builds its search graph once it holds n=500
# items, so a shorter stream would degenerate into buffer appends and measure
# nothing.


@benchmark("neighbors")
def test_knn_classifier_learn(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> None:
        model = neighbors.KNNClassifier(engine=neighbors.SWINN(seed=42))
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("neighbors")
def test_knn_classifier_predict(benchmark) -> None:
    stream = binary_stream(500)
    model = neighbors.KNNClassifier(engine=neighbors.SWINN(seed=42))
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:100]]

    def run() -> None:
        for x in xs:
            model.predict_one(x)

    benchmark(run)
