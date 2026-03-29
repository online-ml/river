from river import evaluate, metrics

from .common import BINARY_CLF_DATASETS, BINARY_CLF_MODELS, get_dataset, get_model


class BinaryClassification:
    """Benchmark binary classification models using progressive validation."""

    timeout = 600
    number = 1
    repeat = 1
    rounds = 1
    min_run_count = 1
    warmup_time = 0

    params = [BINARY_CLF_MODELS, BINARY_CLF_DATASETS]
    param_names = ["model", "dataset"]

    def setup(self, model_name, dataset_name):
        self.model = get_model("binary_clf", model_name)
        # Pre-download remote datasets so download time isn't measured
        dataset = get_dataset("binary_clf", dataset_name)
        next(iter(dataset))
        self.dataset = get_dataset("binary_clf", dataset_name)

    def time_progressive_val_score(self, model_name, dataset_name):
        for _ in evaluate.iter_progressive_val_score(
            dataset=self.dataset,
            model=self.model,
            metric=metrics.Accuracy() + metrics.F1(),
            measure_time=False,
            measure_memory=False,
        ):
            pass
