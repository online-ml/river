"""
A set of sequential optimizers and learning rate schedulers. Also contains loss functions commonly
used in machine learning.
"""
from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .adam import Adam
from .ftrl import FTRLProximal
from .losses import AbsoluteLoss
from .losses import SquaredLoss
from .losses import LogLoss
from .lr_schedule import ConstantLR
from .lr_schedule import LinearDecreaseLR
from .momentum import Momentum
from .nesterov import NesterovMomentum
from .rms_prop import RMSProp
from .vanilla_sgd import VanillaSGD


__all__ = [
    'AdaDelta',
    'AdaGrad',
    'Adam',
    'ConstantLR',
    'FTRLProximal',
    'LinearDecreaseLR',
    'LogLoss',
    'Momentum',
    'NesterovMomentum',
    'RMSProp',
    'AbsoluteLoss',
    'SquaredLoss',
    'VanillaSGD'
]
