from __future__ import annotations

import itertools
import shutil
import zipfile
from urllib import request

from river import stream, utils

from . import base


class BETH(base.RemoteDataset):
    """BETH dataset of system process events.

    This dataset contains labeled host-based telemetry collected from benign and malicious activity.
    The loader uses the process event CSV files and predicts whether an event is labeled as evil.
    DNS logs and the testing CSV are ignored.

    References
    ----------
    [^1]: [BETH dataset on Kaggle](https://www.kaggle.com/katehighnam/beth-dataset)
    [^2]: [Imperial College London data archive](https://data.hpc.imperial.ac.uk/resolve/?doi=9422&file=4&access=)

    """

    def __init__(self):
        super().__init__(
            n_samples=2_666_118,
            n_features=11,
            task=base.BINARY_CLF,
            url="https://data.hpc.imperial.ac.uk/resolve/?doi=9422&file=4&access=",
            size=928_188_305,
            filename=".",
        )

    def download(self, force: bool = False, verbose: bool = True):
        if not force and self.is_downloaded:
            return

        directory = self.path
        directory.mkdir(parents=True, exist_ok=True)
        archive_path = directory.joinpath("full_BETH_dataset.zip")

        with request.urlopen(self.url) as r:
            if verbose:
                meta = r.info()
                try:
                    n_bytes = int(meta["Content-Length"])
                    msg = f"Downloading {self.url} ({utils.pretty.humanize_bytes(n_bytes)})"
                except (KeyError, TypeError):
                    msg = f"Downloading {self.url}"
                print(msg)

            with open(archive_path, "wb") as f:
                shutil.copyfileobj(r, f)

        if verbose:
            print(f"Uncompressing into {directory}")

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(directory)

        archive_path.unlink()

    def _iter(self):  # type: ignore[override]
        converters = {
            "timestamp": float,
            "processId": int,
            "parentProcessId": int,
            "userId": int,
            "eventId": int,
            "argsNum": int,
            "returnValue": int,
            "evil": lambda x: x == "1",
        }

        files = [
            file
            for file in self.path.glob("*.csv")
            if "-dns" not in file.name
            and file.name
            not in {
                "labelled_testing_data.csv",
                "labelled_training_data.csv",
                "labelled_validation_data.csv",
            }
        ]
        return itertools.chain.from_iterable(
            stream.iter_csv(
                file,
                target="evil",
                converters=converters,
                drop=["sus"],
                field_size_limit=1_000_000,
            )
            for file in sorted(files)
        )
