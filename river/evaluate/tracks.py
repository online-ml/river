import typing

from river import datasets, metrics
from river.evaluate.progressive_validation import _progressive_validation


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
    n_samples
        The number of samples that are going to be processed by the track.

    """

    def __init__(self, name: str, dataset, metric, n_samples: int = None):

        if n_samples is None:
            n_samples = dataset.n_samples

        self.name = name
        self.dataset = dataset
        self.metric = metric
        self.n_samples = n_samples

    def run(self, model, n_checkpoints=10):

        # Do the checkpoint logic
        step = self.n_samples // n_checkpoints
        checkpoints = range(0, self.n_samples, step)
        checkpoints = list(checkpoints)[1:] + [self.n_samples]

        # A model might be used in multiple tracks. It's a sane idea to keep things pure and clone
        # the model so that there's no side effects.
        model = model.clone()

        yield from _progressive_validation(
            dataset=self.dataset,
            model=model,
            metric=self.metric,
            checkpoints=iter(checkpoints),
            measure_time=True,
            measure_memory=True,
        )


def load_binary_clf_tracks() -> typing.List[Track]:
    """Return binary classification tracks."""

    return [
        Track(
            name="Phishing",
            dataset=datasets.Phishing(),
            metric=metrics.Accuracy() + metrics.F1(),
        ),
        Track(
            name="Bananas",
            dataset=datasets.Bananas(),
            metric=metrics.Accuracy() + metrics.F1(),
        ),
    ]
