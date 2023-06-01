from methods.base import ChangePointDetector

from numbers import Number
from typing import Dict, List, Tuple, Type, Union
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from math import sqrt
import numpy as np


class Scaler(ABC):
    """
    Abstract base class for feature scaling.

    Subclasses of Scaler should implement the `scale` method.

    Methods:
        scale: Scale the input features.

    Attributes:
        None
    """
    @abstractmethod
    def scale(self, x: Dict[str, float]):
        """
        Scale the input features.

        Args:
            x: Dictionary containing the input features.

        Returns:
            Dict[str, float]: Dictionary containing the scaled features.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Scaler object.

        Returns:
            str: String representation of the Scaler object.
        """
        return self.__class__.__name__


class StandardScaler(Scaler):
    """
    StandardScaler is a Scaler implementation that standardizes features by removing the mean
    and scaling to unit variance.

    Methods:
        scale: Scale the input features using standardization.

    Attributes:
        _counts: Counter object to keep track of feature counts.
        _means: defaultdict object to store the means of each feature.
        _variances: defaultdict object to store the variances of each feature.
    """

    def __init__(self):
        """
        Initialize the StandardScaler object.
        """
        self._counts = Counter()
        self._means = defaultdict(float)
        self._variances = defaultdict(float)

    @abstractmethod
    def non_zero_div(x, y):
        """
        Perform division with handling for zero division.

        Args:
            x: Numerator.
            y: Denominator.

        Returns:
            float: Result of the division.
        """
        try:
            return x/(y)
        except ZeroDivisionError:
            return 0

    def scale(self, x: Dict[str, float]):
        """
        Scale the input features using standardization.

        Args:
            x: Dictionary containing the input features.

        Returns:
            Dict[str, float]: Dictionary containing the scaled features.
        """
        for k, v in x.items():
            self._counts[k] += 1
            prev_mean = self._means[k]
            self._means[k] += (v-prev_mean)/self._counts[k]
            self._variances[k] += (((v-prev_mean) *
                                   (v-self._means[k])) / self._counts[k])
        return {k: StandardScaler.non_zero_div(v-self._means[k], sqrt(self._variances[k])) for k, v in x.items()}


class ECDAS(ChangePointDetector):
    """
    ECDAS (Exponentially Cumulative Distribution-based Algorithm for Stream) is a multivariate change point
    detector that compares the new points with a description of the current distribution.
    A changepoint is detected if the distribution mean goes over a threshold.

    Args:
        num_features: List of numerical features or a single numerical feature.
        cat_features: List of categorical features or a single categorical feature.
        window_size: Size of the sliding window for computing the average.
        num_scaler: Scaler object for scaling numerical features.
        cat_scaler: Scaler object for scaling categorical features.
        custom_start: Custom start index for detecting change points.
        threshold: Threshold value for detecting change points.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        features: Dictionary containing the features used for change point detection.

    Methods:
        update(x, t): Update the change point detector with new data.
        _reset(): Reset the state of the change point detector.
        is_multivariate(): Check if the change point detector is multivariate.
    """

    def __init__(self,
                 num_features: Union[str, List[Union[str, Tuple[str, float]]]] = [
                     "feature1"],
                 cat_features: Union[str,
                                     List[Union[str, Tuple[str, float]]]] = [],
                 window_size: int = 30,
                 num_scaler: Type[Scaler] = None,
                 cat_scaler: Type[Scaler] = None,
                 custom_start: int = None,
                 threshold: float = .9,
                 **kwargs):
        """
        Initialize the ECDAS change point detector.

        Args:
            num_features: List of numerical features or a single numerical feature.
            cat_features: List of categorical features or a single categorical feature.
            window_size: Size of the sliding window for computing the average.
            num_scaler: Scaler object for scaling numerical features.
            cat_scaler: Scaler object for scaling categorical features.
            custom_start: Custom start index for detecting change points.
            threshold: Threshold value for detecting change points.
            **kwargs: Additional keyword arguments to be passed to the base class.
        """
        super().__init__(**kwargs)
        self.features = {0: (None,  # categorical window
                             NumericalWindow(num_features, window_size,
                                             None if not num_scaler else num_scaler()))}
        self._window_size = window_size
        self._num_scaler = num_scaler
        self._custom_start = custom_start if custom_start else window_size
        self.num_feature_count = len(num_features) if isinstance(
            num_features, list) else 1
        self.initialized = False

        # threshold detector
        self.detector = ThresholdChangeDetector(
            mean_threshold=threshold, window_size=window_size, min_samples=window_size)

    def update(self, x, t) -> "ChangePointDetector":
        """
        Update the change point detector with new data.

        Args:
            x: Input data point or dictionary of numerical features.
            t: Time index of the data point.

        Returns:
            self: Updated instance of the change point detector.
        """
        self._change_point_detected = False
        if not isinstance(x, dict):
            x = {"feature1": x}
        if not self.initialized:
            num_features = list(x.keys())
            self.features = {0: (None,  # categorical window
                                 NumericalWindow(num_features, self._window_size,
                                                 None if not self._num_scaler else self._num_scaler()))}
            self.initialized = True
        node_id = False
        f = self.features[node_id if node_id else 0]
        _, num_window = f
        should_score = t+1 >= self._custom_start if not self._custom_start is None \
            else t+1 > self._window_size
        triggered = False
        num_avg, num_features, num_scores = num_window.learn_one(
            x, should_score)
        avg = num_avg  # + cat_features
        if should_score:
            triggered = self.detector.step(avg)
            self._change_point_detected = triggered

        return self

    def _reset(self):
        """
        Reset the state of the change point detector.
        """
        super()._reset()
        self.lookback_values = []

    def is_multivariate(self):
        """
        Check if the change point detector is multivariate.

        Returns:
            bool: True if multivariate, False otherwise.
        """
        return True


class ThresholdChangeDetector:
    """
    ThresholdChangeDetector is a helper class used by ECDAS to detect change points based on a mean threshold.

    Args:
        mean_threshold: Threshold value for detecting change points.
        window_size: Size of the sliding window for computing the mean.
        min_samples: Minimum number of samples required before detecting change points.

    Methods:
        step(x): Perform a step of change point detection.
    """

    def __init__(self, mean_threshold: float, window_size: int = 10, min_samples: int = 100):
        """
        Initialize the ThresholdChangeDetector.

        Args:
            mean_threshold: Threshold value for detecting change points.
            window_size: Size of the sliding window for computing the mean.
            min_samples: Minimum number of samples required before detecting change points.
        """
        assert .0 < mean_threshold < 1.
        self._changepoint = None
        self._min_samples = min_samples
        self._window = deque(maxlen=window_size)
        self._mean_threshold = mean_threshold
        self._min_samples = min_samples
        self._N = 0
        self._mean = 0

    def step(self, x: Number) -> bool:
        """
        Perform a step of change point detection.

        Args:
            x: Input data point.

        Returns:
            bool: True if a change point is detected, False otherwise.
        """
        self._window.append(x)
        self._N += 1
        prev_mean = self._mean
        self._mean += (x - prev_mean) / self._N
        triggered = False
        if self._N > self._min_samples:
            window_mean = sum(self._window) / len(self._window)
            mean_ratio = window_mean / (self._mean + 1e-6)
            if any([mean_ratio > (1. + self._mean_threshold),
                    mean_ratio < (1. - self._mean_threshold)]):
                triggered = True
                self._changepoint = self._N
        return triggered


class Window(ABC):
    """
    Window is an abstract base class representing a sliding window used for change point detection.

    Args:
        features: List of features or a single feature.
        size: Size of the sliding window.
        scaler: Scaler object for scaling the features.
        p: Threshold value for scoring.

    Methods:
        extract_features(x): Extract the relevant features from the input data point.
        _reference_average(): Compute the average of the reference window.
        _update(x): Update the window with a new data point.
    """

    def __init__(self, features: Union[str, List[Union[str, Tuple[str, float]]]], size: int,  scaler: Scaler, p: float = .6):
        """
        Initialize the Window.

        Args:
            features: List of features or a single feature.
            size: Size of the sliding window.
            scaler: Scaler object for scaling the features.
            p: Threshold value for scoring.
        """
        assert .0 < p < 1., "'p' threshold must be between 0 and 1."
        if isinstance(features, (str, tuple)):
            features = [features]
        features = [([*f]+[1.]*max(0, 2-len(f)))[:2] if isinstance(f, (tuple, list)) else (f, 1.)
                    for f in features]
        features, weights = zip(*features)
        self._features = np.asarray(features, dtype=object)
        self._weights = np.asarray(weights)
        self._index = 0
        self._scaler = scaler
        self._p = p
        self.size = size

    def extract_features(self, x: Dict):
        """
        Extract the relevant features from the input data point.

        Args:
            x: Input data point.

        Returns:
            Dict: Extracted features.
        """
        return {k: x[k] if k in x else .0 for k in self._features}

    @abstractmethod
    def _reference_average(self):
        """
        Compute the average of the reference window.

        Returns:
            np.ndarray: Reference window average.
        """
        pass

    @abstractmethod
    def _update(self, x: np.array):
        """
        Update the window with a new data point.

        Args:
            x: New data point.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Window object.

        Returns:
            str: String representation of the Window object.
        """
        return f'{self.__class__.__name__}(features={len(self._features)}, size={self.size}, scaler={self._scaler}, p={self._p})'


class NumericalWindow(Window):
    """
    NumericalWindow is a concrete implementation of the Window class for numerical data.

    Args:
        features: List of features or a single feature.
        size: Size of the sliding window.
        scaler: Scaler object for scaling the features (optional).

    Methods:
        _reference_average(): Compute the average of the reference window.
        _update(x): Update the window with a new data point.
        learn_one(x, should_score=False): Learn from a new data point and compute scores (optional).

    Inherits from:
        Window
    """

    def __init__(self, features: Union[str, List[Union[str, Tuple[str, float]]]], size: int, scaler: Scaler = None):
        """
        Initialize the NumericalWindow.

        Args:
            features: List of features or a single feature.
            size: Size of the sliding window.
            scaler: Scaler object for scaling the features (optional).
        """
        super().__init__(features, size, scaler)
        shape = (size, len(self._features))
        self._reference = np.zeros(shape)
        self._current = np.zeros(shape)

    def _reference_average(self):
        """
        Compute the average of the reference window.

        Returns:
            np.ndarray: Reference window average.
        """
        return np.average(self._reference, axis=0) * self._weights

    def _update(self, x: np.array):
        """
        Update the window with a new data point.

        Args:
            x: New data point.
        """
        if self._index < self._current.shape[0]:
            self._current[self._index] = x
            self._index += 1
        else:
            self._index = 0
            self._reference[:] = self._current
            self._current[self._index] = x
            self._current[self._index+1:] = 0

    # -> mean_score, features, score_per_feature

    def learn_one(self, x: Dict, should_score: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Learn from a new data point and compute scores (optional).

        Args:
            x: Input data point.
            should_score: Flag indicating whether to compute scores.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Mean score, features, and score per feature.
        """
        x = self.extract_features(x)
        if self._scaler:
            x = self._scaler.scale(x)
        x = np.fromiter(x.values(), dtype=np.float64)  # to numpy array
        self._update(x)
        out = np.zeros(len(self._features))
        if not should_score:
            return 0., self._features, out
        ref = self._reference_average()
        loss = Utils.rmse(ref, x) * self._weights
        out[:] = loss
        idx = np.argsort(out)[::-1]
        return np.mean(out, axis=0), self._features[idx], out[idx]


class Utils():
    """
    Utils is a utility class that provides common helper functions.

    Methods:
        rmse(y, x, expand=True): Compute the root mean squared error between two arrays.

    """

    def __init__(self) -> None:
        """
        Initialize the Utils class.
        """
        pass

    def rmse(y: np.ndarray, x: np.ndarray, expand: bool = True):
        """
        Compute the root mean squared error between two arrays.

        Args:
            y: Array containing the ground truth values.
            x: Array containing the predicted values.
            expand: Flag indicating whether to expand the dimensions of y and x (optional).

        Returns:
            np.ndarray: Root mean squared error between y and x.
        """
        if expand:
            y, x = np.expand_dims(y, 0), np.expand_dims(x, 0)
        return np.sqrt(np.mean((y-x)**2, axis=0))
