from __future__ import annotations

import csv
import datetime as dt
import json

from river import datasets


def test_twitch_chat_metadata():
    dataset = datasets.TwitchChat()

    assert dataset.n_samples == 14_720_748
    assert dataset.n_features == 11
    assert dataset.n_classes == 20


def test_twitch_chat_download_size(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVER_DATA", str(tmp_path))
    dataset = datasets.TwitchChat()
    dataset.path.mkdir(parents=True)
    (dataset.path / "partial").write_bytes(b"partial")

    assert not dataset.is_downloaded


def test_twitch_chat_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVER_DATA", str(tmp_path))
    dataset = datasets.TwitchChat()
    data = dataset.path / "data"
    metadata = data / "Meta_Data"
    metadata.mkdir(parents=True)
    with (data / "streamer_2.csv").open("w", newline="") as document_file:
        writer = csv.DictWriter(document_file, fieldnames=["Time", "User", "Message"])
        writer.writeheader()
        writer.writerow({"Time": 3.5, "User": "viewer", "Message": "hello chat"})
    (metadata / "streamer_2_meta.json").write_text(
        json.dumps(
            {
                "user_view_count": 100,
                "user_broadcaster_type": "partner",
                "stream_game_id": "42",
                "stream_type": "live",
                "stream_viewer_count": 50,
                "stream_start_date": "2019-05-30T22:30:30Z",
                "stream_language": "en",
            }
        )
    )

    assert list(dataset._iter()) == [
        (
            {
                "text": "hello chat",
                "time": 3.5,
                "user": "viewer",
                "streamer": "streamer",
                "document": 2,
                "user_view_count": 100,
                "user_broadcaster_type": "partner",
                "stream_type": "live",
                "stream_viewer_count": 50,
                "stream_start_date": dt.datetime(2019, 5, 30, 22, 30, 30, tzinfo=dt.UTC),
                "stream_language": "en",
            },
            "42",
        )
    ]
