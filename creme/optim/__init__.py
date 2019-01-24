"""
To be clear, online gradient descent and stochastic gradient descent are the same thing.

- http://ruder.io/optimizing-gradient-descent/
- https://keras.io/optimisers/
"""
from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .adam import Adam
from .ftrl import FTRLProximal
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
    'Momentum',
    'NesterovMomentum',
    'RMSProp',
    'VanillaSGD'
]
