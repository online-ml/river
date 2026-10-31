from __future__ import annotations

import json

from river import datasets


def test_news_t_metadata():
    dataset = datasets.NewsT()

    assert dataset.n_samples == 11_109
    assert dataset.n_features == 1
    assert dataset.n_classes == 152


def test_news_t_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVER_DATA", str(tmp_path))
    dataset = datasets.NewsT()
    rows = [
        {"Id": "1", "clusterNo": 2, "textCleaned": "first title"},
        {"Id": "2", "clusterNo": 3, "textCleaned": "second title"},
    ]
    dataset.path.parent.mkdir()
    dataset.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert list(dataset._iter()) == [
        ({"text": "first title"}, 2),
        ({"text": "second title"}, 3),
    ]
