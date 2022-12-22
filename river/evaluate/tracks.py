import importlib

from river import datasets, evaluate, metrics


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
    """This track evaluates a model's performance on binary classification tasks.

    Parameter
    ---------
    n_samples
        The number of samples to use for each dataset.

    """
    def __init__(self):
        dsets = [
            getattr(importlib.import_module("river.datasets"), i)()
            for i in importlib.import_module("river.datasets").__all__
            if callable(getattr(importlib.import_module("river.datasets"), i))
            and getattr(importlib.import_module("river.datasets"), i)().task
            == datasets.base.BINARY_CLF
        ]

        super().__init__(
            name="Binary classification",
            datasets=dsets,
            metric=metrics.Accuracy() + metrics.F1(),
        )


class MultiClassClassificationTrack(Track):
    """This track evaluates a model's performance on multi-class classification tasks.

    Parameter
    ---------
    n_samples
        The number of samples to use for each dataset.

    """
    def __init__(self):
        dsets = [
            getattr(importlib.import_module("river.datasets"), i)()
            for i in importlib.import_module("river.datasets").__all__
            if callable(getattr(importlib.import_module("river.datasets"), i))
            and getattr(importlib.import_module("river.datasets"), i)().task
            == datasets.base.MULTI_CLF
        ]

        super().__init__(
            name="Multiclass classification",
            datasets=dsets,
            metric=metrics.Accuracy() + metrics.MicroF1() + metrics.MacroF1(),
        )


class RegressionTrack(Track):
    """This track evaluates a model's performance on regression tasks.

    Parameter
    ---------
    n_samples
        The number of samples to use for each dataset.

    """
    def __init__(self):
        dsets = [
            getattr(importlib.import_module("river.datasets"), i)()
            for i in importlib.import_module("river.datasets").__all__
            if callable(getattr(importlib.import_module("river.datasets"), i))
            and getattr(importlib.import_module("river.datasets"), i)().task == datasets.base.REG
        ]
        super().__init__(
            "Regression",
            datasets=dsets,
            metric=metrics.MAE() + metrics.RMSE() + metrics.R2(),
        )
