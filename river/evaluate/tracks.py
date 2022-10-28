from river import datasets, evaluate, metrics

from .gen import Friedman7k, FriedmanGSG10k, FriedmanLEA10k


class Track:
    """A track evaluate a model's performance.

    The following metrics are recorded:

    - Time, which should be interpreted with wisdom. Indeed time can depend on the architecture
        and local resource situations. Comparison via FLOPS should be preferred.
    - The model's memory footprint.
    - The model's predictive performance on the track's dataset.

    Parameters
    ----------
    name
        The name of the track.
    datasets
        The datasets that compose the track.
    metric
        The metric(s) used to track performance.

    """

    def __init__(self, name: str, datasets, metric):
        self.name = name
        self.datasets = datasets
        self.metric = metric

    def __iter__(self):
        yield from self.datasets

    def run(self, model, dataset, n_checkpoints=10):
        yield from evaluate.iter_progressive_val_score(
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


class MultiClassClassificationTrack(Track):
    def __init__(self):
        super().__init__(
            name="Multiclass classification",
            datasets=[
                datasets.ImageSegments(),
                datasets.Insects(),
                datasets.Keystroke(),
            ],
            metric=metrics.Accuracy() + metrics.MicroF1() + metrics.MacroF1(),
        )


class RegressionTrack(Track):
    def __init__(self):
        super().__init__(
            "Regression",
            datasets=[
                datasets.TrumpApproval(),
                Friedman7k(),
                FriedmanLEA10k(),
                FriedmanGSG10k(),
            ],
            metric=metrics.MAE() + metrics.RMSE() + metrics.R2(),
        )
