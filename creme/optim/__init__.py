"""
A set of sequential optimizers and learning rate schedulers. Also contains loss functions commonly
used in machine learning.
"""
from .ada_bound import AdaBound
from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .adam import Adam
from .base import Optimizer
from .ftrl import FTRLProximal
from .losses import AbsoluteLoss
from .losses import BinaryClassificationLoss
from .losses import Loss
from .losses import MultiClassificationLoss
from .losses import RegressionLoss
from .losses import CauchyLoss
from .losses import CrossEntropy
from .losses import EpsilonInsensitiveHingeLoss
from .losses import HingeLoss
from .losses import LogLoss
from .losses import QuantileLoss
from .losses import SquaredLoss
from .lr_schedule import ConstantLR
from .lr_schedule import InverseScalingLR
from .lr_schedule import OptimalLR
from .mini_batch import MiniBatcher
from .momentum import Momentum
from .nesterov import NesterovMomentum
from .rms_prop import RMSProp
from .sgd import SGD


__all__ = [
    'AbsoluteLoss',
    'AdaBound',
    'AdaDelta',
    'AdaGrad',
    'Adam',
    'BinaryClassificationLoss',
    'CauchyLoss',
    'ConstantLR',
    'CrossEntropy',
    'EpsilonInsensitiveHingeLoss',
    'FTRLProximal',
    'HingeLoss',
    'InverseScalingLR',
    'LogLoss',
    'Loss',
    'MiniBatcher',
    'Momentum',
    'MultiClassificationLoss',
    'NesterovMomentum',
    'OptimalLR',
    'Optimizer',
    'QuantileLoss',
    'RegressionLoss',
    'RMSProp',
    'SGD',
    'SquaredLoss'
]
