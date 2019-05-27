import os
import pytest


@pytest.fixture
def test_path():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def package_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
