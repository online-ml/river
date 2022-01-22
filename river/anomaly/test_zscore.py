import numpy as np
import pytest

from .zscore import Zscore


@pytest.fixture
def warm_up_zs_an_detector() -> Zscore:
    """
    Returns
    -------
    Zscore object
    """
    zs = Zscore(100)
    mean = 10.0
    sd = 2.0

    for _ in range(110):
        v = np.random.normal(mean, sd)
        zs.learn_one(v)
    return zs


@pytest.mark.zscore
def test_zs_anomaly(warm_up_zs_an_detector):
    """
    Parameters
    ----------
    warm_up_zs_an_detector : fixture

    Returns
    -------
    """
    v = 10 + 4 * 2
    assert warm_up_zs_an_detector.score_one(v) >= 1.0


@pytest.mark.zscore
def test_zs_normal(warm_up_zs_an_detector):
    """
    Parameters
    ----------
    warm_up_zs_an_detector : fixture

    Returns
    -------
    """
    v = 10 + 2 * 2
    assert warm_up_zs_an_detector.score_one(v) < 1.0
