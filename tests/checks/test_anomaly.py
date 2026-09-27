from __future__ import annotations

import pytest

from river import base
from river.checks import anomaly as anomaly_checks


class ConstantDetector(base.AnomalyDetector):
    def learn_one(self, x):
        return None

    def score_one(self, x):
        return 0.5


class RankingDetector(base.AnomalyDetector):
    def learn_one(self, x):
        return None

    def score_one(self, x):
        return x["anomalous"]


DATASET = [
    ({"anomalous": 0.0}, False),
    ({"anomalous": 0.0}, False),
    ({"anomalous": 0.0}, False),
    ({"anomalous": 1.0}, True),
    ({"anomalous": 0.0}, False),
]


def test_check_roc_auc_rejects_constant_scores():
    with pytest.raises(AssertionError):
        anomaly_checks.check_roc_auc(ConstantDetector(), DATASET)


def test_check_roc_auc_accepts_ranking_detector():
    anomaly_checks.check_roc_auc(RankingDetector(), DATASET)
