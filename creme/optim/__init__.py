"""
A set of sequential optimizers and learning rate schedulers. Also contains loss functions commonly
used in machine learning.
"""
from . import initializers
from . import losses
from . import schedulers
from .ada_bound import AdaBound
from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .ada_max import AdaMax
from .adam import Adam
from .base import Optimizer
from .ftrl import FTRLProximal
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
    'AdaMax',
    'ConstantLR',
    'CrossEntropy',
    'FTRLProximal',
    'InverseScalingLR',
    'MiniBatcher',
    'Momentum',
    'NesterovMomentum',
    'OptimalLR',
    'Optimizer',
    'RMSProp',
    'SGD'
]
