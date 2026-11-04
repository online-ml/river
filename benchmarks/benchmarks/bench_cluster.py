from river import cluster, datasets, feature_extraction, metrics

from .common import BENCH_SETTINGS


def _extract_text(x):
    return x["text"]


class TextClust:
    timeout = BENCH_SETTINGS["timeout"]
    number = BENCH_SETTINGS["number"]
    repeat = BENCH_SETTINGS["repeat"]
    rounds = BENCH_SETTINGS["rounds"]
    min_run_count = BENCH_SETTINGS["min_run_count"]
    warmup_time = BENCH_SETTINGS["warmup_time"]

    def setup(self):
        self.dataset = datasets.NewsT()
        self.model = (
            _extract_text
            | feature_extraction.BagOfWords()
            | cluster.TextClust(
                fading_factor=0.01,
                tgap=200,
                real_time_fading=False,
                min_weight=1,
                auto_r=True,
                sigma=0.5,
            )
        )

    def _run(self):
        metric = metrics.NormalizedMutualInfo()
        for x, y in self.dataset:
            metric.update(y, self.model.predict_one(x))
            self.model.learn_one(x)
        return metric.get()

    def time_progressive_val_score(self):
        self._run()

    def track_normalized_mutual_info(self):
        return self._run()
