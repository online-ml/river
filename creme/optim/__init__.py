"""
A set of sequential optimizers and learning rate schedulers. Also contains loss functions commonly
used in machine learning.
"""
from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .adam import Adam
from .base import Optimizer
from .ftrl import FTRLProximal
from .losses import AbsoluteLoss
from .losses import BinaryClassificationLoss
from .losses import CauchyLoss
from .losses import CrossEntropy
from .losses import EpsilonInsensitiveHingeLoss
from .losses import HingeLoss
from .losses import LogLoss
from .losses import MultiClassificationLoss
from .losses import RegressionLoss
from .losses import SquaredLoss
from .lr_schedule import ConstantLR
from .lr_schedule import InverseScalingLR
from .lr_schedule import OptimalLR
from .momentum import Momentum
from .nesterov import NesterovMomentum
from .rms_prop import RMSProp
from .vanilla_sgd import VanillaSGD


__all__ = [
    'AbsoluteLoss',
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
    'Momentum',
    'MultiClassificationLoss',
    'NesterovMomentum',
    'OptimalLR',
    'Optimizer',
    'RegressionLoss',
    'RMSProp',
    'SquaredLoss',
    'VanillaSGD'
]
