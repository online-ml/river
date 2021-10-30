from river.linear_model.glm import GLM

from .base import AnomalyDetector


class OneClassSVM(AnomalyDetector, GLM):
    ...
