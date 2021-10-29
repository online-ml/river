from river import base
from river.linear_model.glm import GLM


class OneClassSVM(base.AnomalyDetector, GLM):
    ...
