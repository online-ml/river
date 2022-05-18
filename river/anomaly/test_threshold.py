from river import anomaly
import random


def test_with_supervised_anomaly_detector():
    detector = anomaly.ConstantThresholder(anomaly.UnivariteGaussian(), 2)
    for _ in range(300):
        y = random.gauss(0, 1)
        detector.learn_one(_, y)
    print(detector.anomaly_detector.score_one(_, 3))
    assert detector.score_one(_, 3)
