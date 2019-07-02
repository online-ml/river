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
from .losses.absolute import AbsoluteLoss
from .losses.base import BinaryClassificationLoss
from .losses.base import Loss
from .losses.base import MultiClassificationLoss
from .losses.base import RegressionLoss
from .losses.cauchy import CauchyLoss
from .losses.cross_entropy import CrossEntropy
from .losses.hinge import EpsilonInsensitiveHingeLoss
from .losses.hinge import HingeLoss
from .losses.log_loss import LogLoss
from .losses.quantile import QuantileLoss
from .losses.squared import SquaredLoss
from .lr_schedule import ConstantLR
from .lr_schedule import InverseScalingLR
from .lr_schedule import OptimalLR
from .mini_batch import MiniBatcher
from .momentum import Momentum
from .nesterov import NesterovMomentum
from .rms_prop import RMSProp
from .vanilla_sgd import VanillaSGD


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
    'SquaredLoss',
    'VanillaSGD'
]
