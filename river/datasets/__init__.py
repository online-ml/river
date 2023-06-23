"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from __future__ import annotations

from . import base, synth
from .airline_passengers import AirlinePassengers
from .bananas import Bananas
from .bikes import Bikes
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .higgs import Higgs
from .http import HTTP
from .insects import Insects
from .keystroke import Keystroke
from .malicious_url import MaliciousURL
from .movielens100k import MovieLens100K
from .music import Music
from .phishing import Phishing
from .restaurants import Restaurants
from .segment import ImageSegments
from .sms_spam import SMSSpam
from .smtp import SMTP
from .solar_flare import SolarFlare
from .taxis import Taxis
from .trec07 import TREC07
from .trump_approval import TrumpApproval
from .water_flow import WaterFlow

__all__ = [
    "AirlinePassengers",
    "Bananas",
    "base",
    "Bikes",
    "ChickWeights",
    "CreditCard",
    "Elec2",
    "Higgs",
    "HTTP",
    "ImageSegments",
    "Insects",
    "Keystroke",
    "MaliciousURL",
    "MovieLens100K",
    "Music",
    "Phishing",
    "Restaurants",
    "SMSSpam",
    "SMTP",
    "SolarFlare",
    "synth",
    "Taxis",
    "TREC07",
    "TrumpApproval",
    "WaterFlow",
]


def _docs_overview(print):
    """For website documentation purposes."""

    import collections

    import pandas as pd

    dataset_details = collections.defaultdict(list)

    for dataset_name in __all__:
        if dataset_name in {"base", "synth"}:
            continue
        dataset = eval(dataset_name)()

        details = {
            "Name": f"[{dataset_name}](../datasets/{dataset_name})",
            "Samples": dataset.n_samples,
            "Features": dataset.n_features,
            "Sparse": "✔️" if dataset.sparse else "",
        }

        if dataset.task == base.REG:
            dataset_details[base.REG].append({**details})
        elif dataset.task == base.BINARY_CLF:
            dataset_details[base.BINARY_CLF].append({**details})
        elif dataset.task == base.MULTI_CLF:
            dataset_details[base.MULTI_CLF].append({**details, "Classes": dataset.n_classes})
        elif dataset.task == base.MO_BINARY_CLF:
            dataset_details[base.MO_BINARY_CLF].append({**details, "Outputs": dataset.n_outputs})
        elif dataset.task == base.MO_REG:
            dataset_details[base.MO_REG].append({**details, "Outputs": dataset.n_outputs})
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
