"""Synthetic datasets.

Each synthetic dataset is a stream generator. The benefit of using a generator is that they do not
store the data and each data sample is generated on the fly. Except for a couple of methods,
the majority of these methods are infinite data generators.

"""
from __future__ import annotations

from .agrawal import Agrawal
from .anomaly_sine import AnomalySine
from .concept_drift_stream import ConceptDriftStream
from .friedman import Friedman, FriedmanDrift
from .hyper_plane import Hyperplane
from .led import LED, LEDDrift
from .logical import Logical
from .mixed import Mixed
from .mv import Mv
from .planes_2d import Planes2D
from .random_rbf import RandomRBF, RandomRBFDrift
from .random_tree import RandomTree
from .sea import SEA
from .sine import Sine
from .stagger import STAGGER
from .waveform import Waveform

__all__ = [
    "Agrawal",
    "AnomalySine",
    "ConceptDriftStream",
    "Friedman",
    "FriedmanDrift",
    "Hyperplane",
    "LED",
    "LEDDrift",
    "Logical",
    "Mixed",
    "Mv",
    "Planes2D",
    "RandomRBF",
    "RandomRBFDrift",
    "RandomTree",
    "SEA",
    "Sine",
    "STAGGER",
    "Waveform",
]


def _docs_overview(print):
    """For website documentation purposes."""

    import collections

    import pandas as pd

    from river import datasets

    dataset_details = collections.defaultdict(list)

    for dataset_name in __all__:
        dataset = eval(dataset_name)()

        details = {
            "Name": f"[{dataset_name}](../datasets/synth/{dataset_name})",
            "Features": dataset.n_features,
            "Sparse": "✔️" if dataset.sparse else "",
        }

        if dataset.task == datasets.base.REG:
            dataset_details[datasets.base.REG].append({**details})
        elif dataset.task == datasets.base.BINARY_CLF:
            dataset_details[datasets.base.BINARY_CLF].append({**details})
        elif dataset.task == datasets.base.MULTI_CLF:
            dataset_details[datasets.base.MULTI_CLF].append(
                {**details, "Classes": dataset.n_classes}
            )
        elif dataset.task == datasets.base.MO_BINARY_CLF:
            dataset_details[datasets.base.MO_BINARY_CLF].append(
                {**details, "Outputs": dataset.n_outputs}
            )
        elif dataset.task == datasets.base.MO_REG:
            dataset_details[datasets.base.MO_REG].append({**details, "Outputs": dataset.n_outputs})
        else:
            raise ValueError(f"Unhandled task: {dataset.task}")

    for task, details in dataset_details.items():
        df = pd.DataFrame(details)
        if df.empty:
            continue
        if not df["Sparse"].any():
            df = df.drop(columns=["Sparse"])
        print(f"**{task}**", end="\n\n")
        for int_col in df.select_dtypes(int):
            df[int_col] = df[int_col] = df[int_col].apply(lambda x: f"{int(x):,d}")
        print(df.to_markdown(index=False), end="\n\n")
