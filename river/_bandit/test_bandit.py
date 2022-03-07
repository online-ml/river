from river import base
from river._bandit import Bandit


def test_ranking():
    class DummyMetric(base.Base):
        def __init__(self):
            self.value = None

        def get(self):
            return self.value

        @property
        def bigger_is_better(self):
            return False

    bandit = Bandit(3, DummyMetric())
    bandit.arms[0].metric.value = 0
    bandit.arms[1].metric.value = 1
    bandit.arms[2].metric.value = 2
    assert bandit.ranking == [0, 1, 2]

    bandit.arms[0].metric.value = 2
    bandit.arms[1].metric.value = 1
    bandit.arms[2].metric.value = 0
    assert bandit.ranking == [2, 1, 0]

    bandit.arms[0].metric.value = 0
    bandit.arms[1].metric.value = 2
    bandit.arms[2].metric.value = 1
    assert bandit.ranking == [0, 2, 1]
