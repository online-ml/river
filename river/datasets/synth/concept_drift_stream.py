from __future__ import annotations

import math
import random
import textwrap

from river import datasets
from river.datasets import synth


class ConceptDriftStream(datasets.base.SyntheticDataset):
    """Generates a stream with concept drift.

    A stream generator that adds concept drift or change by joining two
    streams. This is done by building a weighted combination of two pure
    distributions that characterizes the target concepts before and after
    the change.

    The sigmoid function is an elegant and practical solution to define the
    probability that each new instance of the stream belongs to the new
    concept after the drift. The sigmoid function introduces a gradual, smooth
    transition whose duration is controlled with two parameters:

    - $p$, the position of the change.

    - $w$, the width of the transition.

    The sigmoid function at sample $t$ is

    $$f(t) = 1/(1+e^{-4(t-p)/w})$$

    Parameters
    ----------
    stream
        Original stream
    drift_stream
        Drift stream
    seed
        Random seed for reproducibility.
    alpha
        Angle of change used to estimate the width of concept drift change.
        If set, it will override the width parameter. Valid values are in the
        range (0.0, 90.0].
    position
        Central position of the concept drift change.
    width
        Width of concept drift change.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.ConceptDriftStream(
    ...     stream=synth.SEA(seed=42, variant=0),
    ...     drift_stream=synth.SEA(seed=42, variant=1),
    ...     seed=1, position=5, width=2
    ... )

    >>> for x, y in dataset.take(10):
    ...     print(x, y)
    {0: 6.3942, 1: 0.2501, 2: 2.7502} False
    {0: 2.2321, 1: 7.3647, 2: 6.7669} True
    {0: 8.9217, 1: 0.8693, 2: 4.2192} True
    {0: 0.2979, 1: 2.1863, 2: 5.0535} False
    {0: 6.3942, 1: 0.2501, 2: 2.7502} False
    {0: 2.2321, 1: 7.3647, 2: 6.7669} True
    {0: 8.9217, 1: 0.8693, 2: 4.2192} True
    {0: 0.2979, 1: 2.1863, 2: 5.0535} False
    {0: 0.2653, 1: 1.9883, 2: 6.4988} False
    {0: 5.4494, 1: 2.2044, 2: 5.8926} False

    Notes
    -----
    An optional way to estimate the width of the transition $w$ is based on
    the angle $\\alpha$, $w = 1/ tan(\\alpha)$. Since width corresponds to
    the number of samples for the transition, the width is rounded to the
    nearest smaller integer. Notice that larger values of $\\alpha$ result in
    smaller widths. For $\\alpha > 45.0$, the width is smaller than 1 so values
    are rounded to 1 to avoid division by zero errors.

    """

    def __init__(
        self,
        stream: datasets.base.SyntheticDataset | None = None,
        drift_stream: datasets.base.SyntheticDataset | None = None,
        position: int = 5000,
        width: int = 1000,
        seed: int | None = None,
        alpha: float | None = None,
    ):
        if stream is None:
            stream = synth.Agrawal(seed=seed)

        if drift_stream is None:
            drift_stream = synth.Agrawal(seed=seed, classification_function=2)

        # Fairly simple check for consistent number of features
        if stream.n_features != drift_stream.n_features:
            raise AttributeError(
                f"Inconsistent number of features between "
                f"{stream.__class__.__name__} ({stream.n_features}) and "
                f"{drift_stream.__class__.__name__} ({drift_stream.n_features})."
            )
        super().__init__(
            n_features=stream.n_features,
            n_classes=stream.n_classes,
            n_outputs=stream.n_outputs,
            task=stream.task,
        )

        self.n_samples = stream.n_samples

        self.seed = seed
        self.alpha = alpha
        if self.alpha is not None:
            if 0 < self.alpha <= 90.0:
                w = int(1 / math.tan(self.alpha * math.pi / 180))
                self.width = w if w > 0 else 1
            else:
                raise ValueError(
                    f"Invalid alpha value: {alpha}. " f"Valid values are in the range (0.0, 90.0]"
                )
        else:
            self.width = width
        self.position = position
        self.stream = stream
        self.drift_stream = drift_stream

    def __iter__(self):
        rng = random.Random(self.seed)
        stream_generator = iter(self.stream)
        drift_stream_generator = iter(self.drift_stream)
        sample_idx = 0

        while True:
            sample_idx += 1
            v = -4.0 * float(sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + math.exp(v))
            try:
                if rng.random() > probability_drift:
                    x, y = next(stream_generator)
                else:
                    x, y = next(drift_stream_generator)
            except StopIteration:
                break
            yield x, y

    def __repr__(self):
        params = self._get_params()
        l_len_config = max(map(len, params.keys()))
        r_len_config = max(map(len, map(str, params.values())))

        config = "\n\nConfiguration:\n"
        for k, v in params.items():
            if not isinstance(v, datasets.base.SyntheticDataset):
                indent = 0
            else:
                indent = l_len_config + 2
            config += (
                "".join(
                    k.rjust(l_len_config)
                    + "  "
                    + textwrap.indent(str(v).ljust(r_len_config), " " * indent)
                )
                + "\n"
            )

        l_len_prop = max(map(len, self._repr_content.keys()))
        r_len_prop = max(map(len, self._repr_content.values()))

        out = (
            "Synthetic data generator\n\n"
            + "\n".join(
                k.rjust(l_len_prop) + "  " + v.ljust(r_len_prop)
                for k, v in self._repr_content.items()
            )
            + config
        )

        return out
