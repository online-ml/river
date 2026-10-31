from __future__ import annotations

import json

from . import base


class NewsT(base.RemoteDataset):
    """News-T text stream clustering dataset.

    The data contains 11,109 preprocessed news titles assigned to 152 topics. It is derived from
    GoogleNews and reordered into 16 time points so that topics emerge and disappear over time.
    The available feature is the preprocessed title, and the target is the topic identifier.

    References
    ----------
    [^1]: [Dynamic clustering for short text stream based on Dirichlet process](https://doi.org/10.1186/s40537-021-00464-8)
    [^2]: [Textual One-Pass Stream Clustering with Automated Distance Threshold Adaption](https://doi.org/10.1007/978-3-031-21743-2_1)

    """

    def __init__(self):
        super().__init__(
            n_samples=11_109,
            n_features=1,
            n_classes=152,
            task=base.MULTI_CLF,
            url="https://raw.githubusercontent.com/Dennis1989/textClust-experiments/bc5db790bbaf457dafc5c58301bf801afc52e95d/datasets/News-T",
            size=1_053_525,
            filename="news-t.jsonl",
            unpack=False,
        )

    def _iter(self):
        with open(self.path) as stream:
            for line in stream:
                row = json.loads(line)
                yield {"text": row["textCleaned"]}, row["clusterNo"]
