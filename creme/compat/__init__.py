"""Compatibility tools.

This module contains adapters for making ``creme`` estimators compatible with other libraries, and
vice-versa whenever possible. The relevant adapters will only be usable if you have installed the
necessary library. For instance, to use the ``compat.convert_sklearn_to_creme`` function, you have
to install scikit-learn.

"""
import typing

__all__: typing.List[str] = []

try:
    from .sklearn import convert_creme_to_sklearn
    from .sklearn import convert_sklearn_to_creme
    from .sklearn import SKL2CremeClassifier
    from .sklearn import SKL2CremeRegressor
    from .sklearn import Creme2SKLRegressor
    from .sklearn import Creme2SKLClassifier
    from .sklearn import Creme2SKLClusterer
    from .sklearn import Creme2SKLTransformer

    __all__ += [
        'convert_creme_to_sklearn',
        'convert_sklearn_to_creme',
        'Creme2SKLRegressor',
        'Creme2SKLClassifier',
        'Creme2SKLClusterer',
        'Creme2SKLTransformer',
        'SKL2CremeClassifier',
        'SKL2CremeRegressor'
    ]
except ModuleNotFoundError:
    pass

try:
    from .pytorch import PyTorch2CremeRegressor

    __all__ += ['PyTorch2CremeRegressor']
except ModuleNotFoundError:
    pass

