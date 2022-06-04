import typing

from river import datasets, evaluate, metrics


class Track:
    """A track evaluate a model's performance.

    The following metrics are recorded:

    # - FLOPS: floating point operations per second.
    - Time, which should be interpreted with wisdom. Indeed time can depend on the architecture
        and local resource situations. Comparison via FLOPS should be preferred.
    - The model's memory footprint.
    - The model's predictive performance on the track's dataset.

    Parameters
    ----------
    name
        The name of the track.
    dataset
        The dataset from which samples will be retrieved. A slice must be used if the dataset
        is a data generator.
    metric
        The metric(s) used to track performance.

    """

    def __init__(self, name: str, datasets, metric):
        self.name = name
        self.datasets = datasets
        self.metric = metric

    def run(self, model, n_checkpoints=10):
        for dataset in self.datasets:
            yield evaluate.iter_progressive_val_score(
                dataset=dataset,
                model=model.clone(),
                metric=self.metric.clone(),
                step=dataset.n_samples // n_checkpoints,
                measure_time=True,
                measure_memory=True,
            )



class BinaryClassificationTrack(Track):
    def __init__(self):
        super().__init__(
            name="Binary classification",
            datasets=[datasets.Phishing(), datasets.Bananas()],
            metric=metrics.Accuracy() + metrics.F1(),
        )
