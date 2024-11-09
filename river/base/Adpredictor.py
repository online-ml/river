from __future__ import annotations

import logging
from collections import defaultdict, namedtuple

import numpy as np

from river import base

logger = logging.getLogger(__name__)


class AdPredictor(base.Classifier):
    config = namedtuple("config", ["beta", "prior_probability", "epsilon", "num_features"])

    def __init__(self, beta=0.1, prior_probability=0.5, epsilon=0.1, num_features=10):
        self.beta = beta
        self.prior_probability = prior_probability
        self.epsilon = epsilon
        self.num_features = num_features
        self.weights = defaultdict(lambda: {"mean": 0.0, "variance": 1.0})
        self.bias_weight = self.prior_bias_weight()

    def prior_bias_weight(self):
        return np.log(self.prior_probability / (1 - self.prior_probability)) / self.beta

    def _active_mean_variance(self, features):
        total_mean = sum(self.weights[f]["mean"] for f in features) + self.bias_weight
        total_variance = sum(self.weights[f]["variance"] for f in features) + self.beta**2
        return total_mean, total_variance

    def predict_one(self, x):
        features = x.keys()
        total_mean, total_variance = self._active_mean_variance(features)
        return 1 / (1 + np.exp(-total_mean / np.sqrt(total_variance)))

    def learn_one(self, x, y):
        features = x.keys()
        y = 1 if y else -1  # Map label to ±1 for binary classification
        total_mean, total_variance = self._active_mean_variance(features)
        v, w = self.gaussian_corrections(y * total_mean / np.sqrt(total_variance))

        for feature in features:
            mean = self.weights[feature]["mean"]
            variance = self.weights[feature]["variance"]

            mean_delta = y * variance / np.sqrt(total_variance) * v
            variance_multiplier = 1.0 - variance / total_variance * w

            # Update weight
            self.weights[feature]["mean"] = mean + mean_delta
            self.weights[feature]["variance"] = variance * variance_multiplier

    def gaussian_corrections(self, score):
        """Compute Gaussian corrections for Bayesian update."""
        cdf = 1 / (1 + np.exp(-score))
        pdf = np.exp(-0.5 * score**2) / np.sqrt(2 * np.pi)
        v = pdf / cdf
        w = v * (v + score)
        return v, w

    def _apply_dynamics(self, weight):
        prior_variance = 1.0
        adjusted_variance = (
            weight["variance"]
            * prior_variance
            / ((1.0 - self.epsilon) * prior_variance + self.epsilon * weight["variance"])
        )
        adjusted_mean = adjusted_variance * (
            (1.0 - self.epsilon) * weight["mean"] / weight["variance"]
            + self.epsilon * 0 / prior_variance
        )
        return {"mean": adjusted_mean, "variance": adjusted_variance}

    def __str__(self):
        return "AdPredictor"
