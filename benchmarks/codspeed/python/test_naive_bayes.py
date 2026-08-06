from marks import benchmark
from workloads import N_PREDICT, multiclass_stream, text_stream

from river import feature_extraction, naive_bayes


@benchmark("naive_bayes")
def test_gaussian_nb_learn(benchmark) -> None:
    stream = multiclass_stream()

    def run() -> None:
        model = naive_bayes.GaussianNB()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@benchmark("naive_bayes")
def test_gaussian_nb_predict(benchmark) -> None:
    stream = multiclass_stream()
    model = naive_bayes.GaussianNB()
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_proba_one(x)

    benchmark(run)


@benchmark("naive_bayes")
def test_multinomial_nb_learn(benchmark) -> None:
    bow = feature_extraction.BagOfWords()
    stream = [(bow.transform_one(text), i % 10) for i, text in enumerate(text_stream())]

    def run() -> None:
        model = naive_bayes.MultinomialNB()
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
