from __future__ import annotations

import csv
import datetime as dt
import json

from . import base


class TwitchChat(base.RemoteDataset):
    """Twitch livestream chat messages.

    The data contains 14,720,748 messages from 1,951 livestream sessions by 666 streamers.
    Each message has an anonymized user identifier, a relative timestamp, and session metadata.
    The target is the identifier of the game streamed during the session.

    References
    ----------
    [^1]: [TwitchChat: A Dataset for Exploring Livestream Chat](https://doi.org/10.1609/aiide.v16i1.7439)
    [^2]: [Original dataset](https://osf.io/39ev7/)

    """

    def __init__(self):
        super().__init__(
            n_samples=14_720_748,
            n_features=11,
            n_classes=20,
            task=base.MULTI_CLF,
            url="https://osf.io/download/bys56/",
            size=1_172_317_111,
            filename="data_set",
            archive_name="data_set.zip",
        )

    def _iter(self):
        data = self.path / "data"
        for document_path in sorted(data.glob("*.csv")):
            streamer, document = document_path.stem.rsplit("_", maxsplit=1)
            metadata_path = data / "Meta_Data" / f"{document_path.stem}_meta.json"
            with metadata_path.open() as metadata_file:
                metadata = json.load(metadata_file)
            session = {
                "streamer": streamer,
                "document": int(document),
                "user_view_count": metadata["user_view_count"],
                "user_broadcaster_type": metadata["user_broadcaster_type"],
                "stream_type": metadata["stream_type"],
                "stream_viewer_count": metadata["stream_viewer_count"],
                "stream_start_date": dt.datetime.fromisoformat(metadata["stream_start_date"]),
                "stream_language": metadata["stream_language"],
            }
            with document_path.open(newline="") as document_file:
                for row in csv.DictReader(document_file):
                    yield (
                        {
                            "text": row["Message"],
                            "time": float(row["Time"]),
                            "user": row["User"],
                            **session,
                        },
                        metadata["stream_game_id"],
                    )
