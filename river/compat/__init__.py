"""Compatibility tools.

This module contains adapters for making `river` estimators compatible with other libraries, and
vice-versa whenever possible. The relevant adapters will only be usable if you have installed the
necessary library. For instance, you have to install scikit-learn in order to use the
`compat.convert_sklearn_to_river` function.

"""
import typing

__all__: typing.List[str] = []

try:
    from .river_to_sklearn import convert_river_to_sklearn
    from .river_to_sklearn import River2SKLRegressor
    from .river_to_sklearn import River2SKLClassifier
    from .river_to_sklearn import River2SKLClusterer
    from .river_to_sklearn import River2SKLTransformer
    from .sklearn_to_river import convert_sklearn_to_river
    from .sklearn_to_river import SKL2RiverClassifier
    from .sklearn_to_river import SKL2RiverRegressor

    __all__ += [
        "convert_river_to_sklearn",
        "convert_sklearn_to_river",
        "River2SKLRegressor",
        "River2SKLClassifier",
        "River2SKLClusterer",
        "River2SKLTransformer",
        "SKL2RiverClassifier",
        "SKL2RiverRegressor",
    ]
except ModuleNotFoundError:
    pass

try:
    from .pytorch import PyTorch2RiverRegressor

    __all__ += ["PyTorch2RiverRegressor"]
except ModuleNotFoundError:
    pass
