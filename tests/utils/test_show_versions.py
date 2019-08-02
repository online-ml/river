import pytest

import sys

from skmultiflow.utils._show_versions import _get_deps_info
from skmultiflow.utils import show_versions


def test_get_deps_info():
    info = _get_deps_info()
    expected_keys = [
        "pip",
        "setuptools",
        "skmultiflow",
        "sklearn",
        "numpy",
        "scipy",
        "sortedcontainers",
        "matplotlib",
        "pandas",
    ]
    assert set(info.keys()) == set(expected_keys)


@pytest.mark.skipif(sys.platform == "win32", reason="does not work on windows")
def test_show_versions(capsys):
    show_versions()
    captured = capsys.readouterr()
    assert captured.out.startswith("\nSystem:\n")

    show_versions(github=True)
    captured = capsys.readouterr()
    assert captured.out.startswith("<details><summary>System, Dependencies</summary>\n\n"
                                   "**System Information**\n\n")
