from river import stream

from . import base


class ImageSegments(base.FileDataset):
    """Image segments classification.

    This dataset contains features that describe image segments into 7 classes: brickface, sky,
    foliage, cement, window, path, and grass.

    References
    ----------
    [^1]: [UCI page](https://archive.ics.uci.edu/ml/datasets/Statlog+(Image+Segmentation))

    """

    def __init__(self):
        super().__init__(
            n_samples=2310,
            n_features=18,
            task=base.MULTI_CLF,
            filename="segment.csv.zip",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="category",
            converters={
                "region-centroid-col": int,
                "region-centroid-row": int,
                "short-line-density-5": float,
                "short-line-density-2": float,
                "vedge-mean": float,
                "vegde-sd": float,
                "hedge-mean": float,
                "hedge-sd": float,
                "intensity-mean": float,
                "rawred-mean": float,
                "rawblue-mean": float,
                "rawgreen-mean": float,
                "exred-mean": float,
                "exblue-mean": float,
                "exgreen-mean": float,
                "value-mean": float,
                "saturation-mean": float,
                "hue-mean": float,
            },
        )
