from abc import ABC, abstractmethod

#import abc
import numpy as np
import math
import utils

#from pysad.utils import _iterate


class BaseTransformer(ABC):
    """Base class for transforming methods.
    """

    def __init__(self, output_dims):
        self.output_dims = output_dims

    @abstractmethod
    def learn_one(self, X):
        """Fits particular (next) timestep's features to train the transformer.
        Args:
            X (np.float array of shape (num_components,)): Input feature vector.
        Returns:
            object: self.
        """
        pass

    @abstractmethod
    def transform_one(self, X):
        """Transforms particular (next) timestep's vector.
        Args:
            X (np.float array of shape (num_features,)): Input feature vector.
        Returns:
            transformed_X (np.float array of shape (num_components,)): Projected feature vector.
        """
        pass

    def learn_transform_one(self, X):
        """Shortcut method that iteratively applies fit_partial and transform_partial, respectively.
        Args:
            X (np.float array of shape (num_components,)): Input feature vector.
        Returns:
            transformed_X (np.float array of shape (num_components,)): Projected feature vector.
        """
        return self.learn_one(X).transform_one(X)

    def transform(self, X):
        """Shortcut method that iteratively applies transform_partial to all instances in order.
        Args:
            X (np.float array of shape (num_instances, num_features)): Input feature vectors.
        Returns:
            np.float array of shape (num_instances, num_components): Projected feature vectors.
        """
        output_dims = self.output_dims if self.output_dims > 0 else X.shape[1]
        transformed_X = np.empty((X.shape[0], output_dims), dtype=np.float)
        for i, (xi, _) in enumerate(_iterate(X)):
            transformed_X[i] = self.transform_one(xi)

        return transformed_X

    def fit(self, X):
        """Shortcut method that iteratively applies fit_partial to all instances in order.
        Args:
            X (np.float array of shape (num_instances, num_features)): Input feature vectors.
        Returns:
            object: The fitted transformer
        """
        for xi in _iterate(X):
            self.learn_one(xi)

        return self

    def learn_transform(self, X):
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.
        Args:
            X (np.float array of shape (num_instances, num_components)): Input feature vectors.
        Returns:
            np.float array of shape (num_instances, num_components): Projected feature vectors.
        """
        output_dims = self.output_dims if self.output_dims > 0 else X.shape[1]
        transformed_X = np.empty((X.shape[0], output_dims), dtype=np.float)
        for i, (xi, _) in enumerate(_iterate(X)):
            transformed_X[i] = self.learn_transform_one(xi)

        return transformed_X