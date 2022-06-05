from river import base, datasets, evaluate, metrics, synth


class WrappedGenerator(datasets.base.Dataset):
    def __init__(self, n_samples, n_features, task, gen):
        super().__init__(
            filename="generator-based",
            n_samples=n_samples,
            n_features=n_features,
            task=task,
        )

        self._gen = gen

    def __iter__(self):
        for i, (x, y) in enumerate(self._gen):
            if i == self.n_samples:
                break
            yield x, y


class Friedman7k(WrappedGenerator):
    """Sample from the stationary version of the Friedman dataset.

    This sample contains 10k instances sampled from the Friedman generator.

    """

    def __init__(self):
        super().__init__(
            n_samples=7000,
            n_features=10,
            task=base.REGRESSION,
            gen=synth.Friedman(seed=42),
        )


class FriedmanLEA10k(WrappedGenerator):
    """Sample from the FriedmanLEA generator.

    This sample contains 10k instances sampled from the Friedman generator and presents
    local-expanding abrupt concept drifts that locally affect the data and happen after
    2k, 5k, and 8k instances.

    """

    def __init__(self):
        super().__init__(
            n_samples=10000,
            n_features=10,
            task=base.REGRESSION,
            gen=synth.FriedmanDrift(
                drift_type="lea", position=(2000, 5000, 8000), seed=42
            ),
        )


class FriedmanGSG10k(WrappedGenerator):
    """Sample from the FriedmanGSG generator.

    This sample contains 10k instances sampled from the Friedman generator and presents
    global and slow gradual concept drifts that affect the data and happen after
    3.5k and 7k instances. The transition window between different concepts has a length of
    1k instances.

    """

    def __init__(self):
        super().__init__(
            n_samples=10000,
            n_features=10,
            task=base.REGRESSION,
            gen=synth.FriedmanDrift(
                drift_type="gsg", position=(3500, 7000), transition_window=1000, seed=42
            ),
        )


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


class RegressionTrack(Track):
    def __init__(self):
        super().__init__(
            "Single-target Regression",
            datasets=[
                datasets.TrumpApproval(),
                Friedman7k(),
                FriedmanLEA10k(),
                FriedmanGSG10k(),
            ],
            metric=metrics.MAE() + metrics.RMSE() + metrics.R2(),
        )
