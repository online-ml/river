from river import stream

from . import base


class Phishing(base.FileDataset):
    """Phishing websites.

    This dataset contains features from web pages that are classified as phishing or not.

    References
    ----------
    [^1]: [UCI page](http://archive.ics.uci.edu/ml/datasets/Website+Phishing)

    """

    def __init__(self):
        super().__init__(
            n_samples=1250,
            n_features=9,
            task=base.BINARY_CLF,
            filename="phishing.csv.gz",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="is_phishing",
            converters={
                "empty_server_form_handler": float,
                "popup_window": float,
                "https": float,
                "request_from_other_domain": float,
                "anchor_from_other_domain": float,
                "is_popular": float,
                "long_url": float,
                "age_of_domain": int,
                "ip_in_url": int,
                "is_phishing": lambda x: x == "1",
            },
        )
