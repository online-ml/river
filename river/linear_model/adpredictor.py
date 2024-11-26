import math

from collections import *

from river.base.classifier import Classifier


def default_mean():
    return 0.0

def default_variance():
    return 1.0


class AdPredictor(Classifier):
    """
    AdPredictor is a machine learning algorithm designed to predict the probability of user
    clicks on online advertisements. This algorithm plays a crucial role in computational advertising, where predicting
    click-through rates (CTR) is essential for optimizing ad placements and maximizing revenue.
    Parameters
    ----------
    beta (float, default=0.1):
    A smoothing parameter that regulates the weight updates. Smaller values allow for finer updates,
    while larger values can accelerate convergence but may risk instability.
    prior_probability (float, default=0.5):
    The initial estimate rate. This value sets the bias weight, influencing the model's predictions
    before observing any data.

    epsilon (float, default=0.1):
    A variance dynamics parameter that controls how the model balances prior knowledge and learned information.
    Larger values prioritize prior knowledge, while smaller values favor data-driven updates.

    num_features (int, default=10):
    The maximum number of features the model can handle. This parameter affects scalability and efficiency,
    especially for high-dimensional data.

    Attributes
    ----------
    weights (defaultdict):
    A dictionary where each feature key maps to a dictionary containing:

    mean (float): The current estimate of the feature's weight.
    variance (float): The uncertainty associated with the weight estimate.

    bias_weight (float):
    The weight corresponding to the model bias, initialized using the prior_probability.
    This attribute allows the model to make predictions even when no features are active.

    Examples:
    ----------

    >>> from river.linear_model import AdPredictor
    >>> adpredictor = AdPredictor(beta=0.1, prior_probability=0.5, epsilon=0.1, num_features=5)
    >>> data = [({"feature1": 1, "feature2": 1}, 1),({"feature1": 1, "feature3": 1}, 0),({"feature2": 1, "feature4": 1}, 1),({"feature1": 1, "feature2": 1, "feature3": 1}, 0),({"feature4": 1, "feature5": 1}, 1),]
    >>> def train_and_test(model, data):
    ...    for x, y in data:
    ...        pred_before = model.predict_one(x)
    ...        model.learn_one(x, y)
    ...        pred_after = model.predict_one(x)
    ...        print(f"Features: {x} | True label: {y} | Prediction before training: {pred_before:.4f} | Prediction after training: {pred_after:.4f}")

    >>> train_and_test(adpredictor, data)
    Features: {'feature1': 1, 'feature2': 1} | True label: 1 | Prediction before training: 0.5000 | Prediction after training: 0.7230
    Features: {'feature1': 1, 'feature3': 1} | True label: 0 | Prediction before training: 0.6065 | Prediction after training: 0.3650
    Features: {'feature2': 1, 'feature4': 1} | True label: 1 | Prediction before training: 0.6065 | Prediction after training: 0.7761
    Features: {'feature1': 1, 'feature2': 1, 'feature3': 1} | True label: 0 | Prediction before training: 0.5455 | Prediction after training: 0.3197
    Features: {'feature4': 1, 'feature5': 1} | True label: 1 | Prediction before training: 0.5888 | Prediction after training: 0.7699

    """



    def __init__(self, beta=0.1, prior_probability=0.5, epsilon=0.1, num_features=10):
        # Initialization of model parameters
        self.beta = beta
        self.prior_probability = prior_probability
        self.epsilon = epsilon
        self.num_features = num_features
        # Initialize weights as a defaultdict for each feature, with mean and variance attributes

        self.means = defaultdict(default_mean)
        self.variances = defaultdict(default_variance)


        # Initialize bias weight based on prior probability
        self.bias_weight = self.prior_bias_weight()

    def prior_bias_weight(self):
        # Calculate initial bias weight using prior probability

        return math.log(self.prior_probability / (1 - self.prior_probability)) / self.beta

    def _active_mean_variance(self, features):
        """_active_mean_variance(features) (method):
        Computes the cumulative mean and variance for all active features in a sample,
        including the bias. This is crucial for making predictions."""
        # Calculate total mean and variance for all active features

        total_mean = sum(self.means[f] for f in features) + self.bias_weight
        total_variance = sum(self.variances[f] for f in features) + self.beta ** 2
        return total_mean, total_variance

    def predict_one(self, x):
        # Generate a probability prediction for one sample
        features = x.keys()
        total_mean, total_variance = self._active_mean_variance(features)
        # Sigmoid function for probability prediction based on Gaussian distribution
        return 1 / (1 + math.exp(-total_mean / math.sqrt(total_variance)))

    def learn_one(self, x, y):
        # Online learning step to update the model with one sample
        features = x.keys()
        y = 1 if y else -1
        total_mean, total_variance = self._active_mean_variance(features)
        v, w = self.gaussian_corrections(y * total_mean / math.sqrt(total_variance))

        # Update mean and variance for each feature in the sample
        for feature in features:
            mean = self.means[feature]
            variance = self.variances[feature]

            mean_delta = y * variance / math.sqrt(total_variance) * v # Update mean
            variance_multiplier = 1.0 - variance / total_variance * w # Update variance

            # Update weight
            self.means[feature] = mean + mean_delta
            self.variances[feature]= variance * variance_multiplier


    def gaussian_corrections(self, score):
        """gaussian_corrections(score) (method):
        Implements Bayesian update corrections using the Gaussian probability density function (PDF)
        and cumulative density function (CDF)."""
        # CDF calculation for Gaussian correction
        cdf = 1 / (1 + math.exp(-score))
        pdf = math.exp(-0.5 * score**2) / math.sqrt(2 * math.pi)  # PDF calculation
        v = pdf / cdf  # Correction factor for mean update
        w = v * (v + score)  # Correction factor for variance update
        return v, w

    def _apply_dynamics(self, weight):
        """_apply_dynamics(weight) (method):
        Regularizes the variance of a feature weight using a combination of prior variance and learned variance.
        This helps maintain a balance between prior beliefs and observed data."""
        # Apply variance dynamics for regularization
        prior_variance = 1.0
        # Adjust variance to manage prior knowledge and current learning balance
        adjusted_variance = (
                weight["variance"]
                * prior_variance
                / ((1.0 - self.epsilon) * prior_variance + self.epsilon * weight["variance"]))
        # Adjust mean based on the dynamics, balancing previous and current knowledge
        adjusted_mean = adjusted_variance * (
                (1.0 - self.epsilon) * weight["mean"] / weight["variance"]
                + self.epsilon * 0 / prior_variance
        )
        return {"mean": adjusted_mean, "variance": adjusted_variance}


