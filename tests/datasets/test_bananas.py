from __future__ import annotations

import typing
import zipfile

import pytest

from river import datasets

if typing.TYPE_CHECKING:
    import pathlib


def test_bundled_bananas_can_be_read() -> None:
    assert list(datasets.Bananas().take(2)) == [
        ({"1": 1.617466, "2": -0.919233}, False),
        ({"1": -1.394669, "2": 1.094125}, False),
    ]


def test_bananas_reads_its_data_member_instead_of_metadata(tmp_path: pathlib.Path) -> None:
    dataset = datasets.Bananas()
    dataset.directory = tmp_path
    with zipfile.ZipFile(dataset.path, "w") as archive:
        archive.writestr("__MACOSX/._banana.all.txt", b"metadata")
        archive.writestr("banana.all.txt", "-1 1:1.0 2:2.0\n1 1:3.0 2:4.0\n")

    assert list(dataset) == [
        ({"1": 1.0, "2": 2.0}, False),
        ({"1": 3.0, "2": 4.0}, True),
    ]


def test_bananas_requires_its_named_data_member(tmp_path: pathlib.Path) -> None:
    dataset = datasets.Bananas()
    dataset.directory = tmp_path
    with zipfile.ZipFile(dataset.path, "w") as archive:
        archive.writestr("other.txt", "1 1:1.0\n")

    with pytest.raises(KeyError, match="banana.all.txt"):
        list(dataset)
