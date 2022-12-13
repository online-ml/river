from __future__ import annotations

import random
import warnings

from river import datasets


class RandomRBF(datasets.base.SyntheticDataset):
    """Random Radial Basis Function generator.

    Produces a radial basis function stream. A number of centroids, having a
    random central position, a standard deviation, a class label and weight
    are generated. A new sample is created by choosing one of the centroids at
    random, taking into account their weights, and offsetting the attributes
    in a random direction from the centroid's center. The offset length is
    drawn from a Gaussian distribution.

    This process will create a normally distributed hypersphere of samples on
    the surrounds of each centroid.

    Parameters
    ----------
    seed_model
        Model's random seed to generate centroids.
    seed_sample
        Sample's random seed.
    n_classes
        The number of class labels to generate.
    n_features
        The number of numerical features to generate.
    n_centroids
        The number of centroids to generate.

    Examples
    --------
    >>> from river.datasets import synth
    >>>
    >>> dataset = synth.RandomRBF(seed_model=42, seed_sample=42,
    ...                           n_classes=4, n_features=4, n_centroids=20)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 1.0989, 1: 0.3840, 2: 0.7759, 3: 0.6592} 2
    {0: 0.2366, 1: 1.3233, 2: 0.5691, 3: 0.2083} 0
    {0: 1.3540, 1: -0.3306, 2: 0.1683, 3: 0.8865} 0
    {0: 0.2585, 1: -0.2217, 2: 0.4739, 3: 0.6522} 0
    {0: 0.1295, 1: 0.5953, 2: 0.1774, 3: 0.6673} 1

    """

    def __init__(
        self,
        seed_model: int | None = None,
        seed_sample: int | None = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
    ):
        super().__init__(
            n_features=n_features,
            n_classes=n_classes,
            n_outputs=1,
            task=datasets.base.MULTI_CLF,
        )
        self.seed_sample = seed_sample
        self.seed_model = seed_model
        self.n_num_features = n_features
        self.n_centroids = n_centroids
        self.centroids: list = []
        self.centroid_weights: list = []
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._generate_centroids()
        rng_sample = random.Random(self.seed_sample)

        while True:
            x, y = self._generate_sample(rng_sample)
            yield x, y

    def _generate_sample(self, rng_sample: random.Random):
        idx = random_index_based_on_weights(self.centroid_weights, rng_sample)
        current_centroid = self.centroids[idx]
        att_vals = dict()
        magnitude = 0.0
        for i in range(self.n_features):
            att_vals[i] = (rng_sample.random() * 2.0) - 1.0
            magnitude += att_vals[i] * att_vals[i]
        magnitude = magnitude**0.5
        desired_mag = rng_sample.gauss(0, 1) * current_centroid.std_dev
        scale = desired_mag / magnitude
        x = {i: current_centroid.centre[i] + att_vals[i] * scale for i in range(self.n_features)}
        y = current_centroid.class_label
        return x, y

    def _generate_centroids(self):
        """Generates centroids

        Sequentially creates all the centroids, choosing at random a center,
        a label, a standard deviation and a weight.

        """
        rng_model = random.Random(self.seed_model)
        self.centroids = []
        self.centroid_weights = []
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(rng_model.random())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = rng_model.randint(0, self.n_classes - 1)
            self.centroids[i].std_dev = rng_model.random()
            self.centroid_weights.append(rng_model.random())


class RandomRBFDrift(RandomRBF):
    """Random Radial Basis Function generator with concept drift.

    This class is an extension from the `RandomRBF` generator. Concept drift
    can be introduced in instances of this class.

    The drift is created by adding a "speed" to certain centroids. As the
    samples are generated each of the moving centroids' centers is
    changed by an amount determined by its speed.

    Parameters
    ----------
    seed_model
        Model's random seed to generate centroids.
    seed_sample
        Sample's random seed.
    n_classes
        The number of class labels to generate.
    n_features
        The number of numerical features to generate.
    n_centroids
        The number of centroids to generate.
    change_speed
        The concept drift speed.
    n_drift_centroids
        The number of centroids that will drift.

    Examples
    --------
    >>> from river.datasets import synth
    >>>
    >>> dataset = synth.RandomRBFDrift(seed_model=42, seed_sample=42,
    ...                                n_classes=4, n_features=4, n_centroids=20,
    ...                                change_speed=0.87, n_drift_centroids=10)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 1.0989, 1: 0.3840, 2: 0.7759, 3: 0.6592} 2
    {0: 1.1496, 1: 1.9014, 2: 1.5393, 3: 0.3210} 0
    {0: 0.7146, 1: -0.2414, 2: 0.8933, 3: 1.6633} 0
    {0: 0.3797, 1: -0.1027, 2: 0.8717, 3: 1.1635} 0
    {0: 0.1295, 1: 0.5953, 2: 0.1774, 3: 0.6673} 1


    """

    def __init__(
        self,
        seed_model: int | None = None,
        seed_sample: int | None = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
        change_speed: float = 0.0,
        n_drift_centroids: int = 50,
    ):
        super().__init__(
            seed_model=seed_model,
            seed_sample=seed_sample,
            n_classes=n_classes,
            n_features=n_features,
            n_centroids=n_centroids,
        )
        self.change_speed = change_speed
        if n_drift_centroids <= n_centroids:
            self.n_drift_centroids = n_drift_centroids
        else:
            warnings.warn(
                f"n_drift_centroids ({n_drift_centroids}) can not be larger than"
                f"n_centroids ({n_centroids}). Will use n_centroids instead."
            )
            self.n_drift_centroids = n_centroids
        self.centroid_speed = None

    def __iter__(self):
        self._generate_centroids()
        rng_sample = random.Random(self.seed_sample)

        while True:
            # Move centroids
            for i in range(self.n_drift_centroids):
                for j in range(self.n_features):
                    self.centroids[i].centre[j] += self.centroid_speed[i][j] * self.change_speed

                    if (self.centroids[i].centre[j] > 1) or (self.centroids[i].centre[j] < 0):
                        self.centroids[i].centre[j] = 1 if (self.centroids[i].centre[j] > 1) else 0
                        self.centroid_speed[i][j] = -self.centroid_speed[i][j]

            x, y = self._generate_sample(rng_sample)
            yield x, y

    def _generate_centroids(self):
        """Generates centroids

        The centroids are generated just as it is done in the parent class,
        an extra step is taken to introduce drift, if there is any.

        To configure the drift, random offset speeds are chosen for
        `n_drift_centroids` centroids. Finally, the speed is normalized.

        """
        super()._generate_centroids()
        rng_model = random.Random(self.seed_model)
        self.centroid_speed = []

        for i in range(self.n_drift_centroids):
            rand_speed = [0] * self.n_features
            norm_speed = 0.0

            for j in range(self.n_features):
                rand_speed[j] = rng_model.random()
                norm_speed += rand_speed[j] * rand_speed[j]

            norm_speed = norm_speed**0.5

            for j in range(self.n_features):
                rand_speed[j] /= norm_speed

            self.centroid_speed.append(rand_speed)


class Centroid:
    """Class that stores a centroid's attributes."""

    def __init__(self):
        self.centre = None
        self.class_label = None
        self.std_dev = None


def random_index_based_on_weights(weights: list, rng: random.Random):
    """Generate a random index, based on index weights and a random number generator.

    Parameters
    ----------
    weights
        The weights of the centroid's indexes.

    rng
        Random number generator instance.

    Returns
    -------
    int
        The generated index.

    """
    prob_sum = sum(weights)
    val = rng.random() * prob_sum
    index = 0
    sum_value = 0.0
    while (sum_value <= val) & (index < len(weights)):
        sum_value += weights[index]
        index += 1
    return index - 1
