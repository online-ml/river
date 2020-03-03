"""Linear models."""
from .alma import ALMAClassifier
from .glm import LinearRegression
from .glm import LogisticRegression
from .pa import PAClassifier
from .pa import PARegressor
from .softmax import SoftmaxRegression


__all__ = [
    'ALMAClassifier',
    'LinearRegression',
    'LogisticRegression',
    'PAClassifier',
    'PARegressor',
    'SoftmaxRegression'
]
