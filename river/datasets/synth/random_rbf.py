import numpy as np
import warnings

from .. import base
from river.utils.skmultiflow_utils import check_random_state


class RandomRBF(base.SyntheticDataset):
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
        Model's seed to generate centroids
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    seed_sample
        Sample's seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_classes
        The number of class labels to generate.
    n_features
        The number of numerical features to generate.
    n_centroids
        The number of centroids to generate.

    Examples
    --------
    >>> from river import synth
    >>>
    >>> dataset = synth.RandomRBF(seed_model=42, seed_sample=42,
    ...                           n_classes=4, n_features=4, n_centroids=20)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 0.9518, 1: 0.5263, 2: 0.2509, 3: 0.4177} 0
    {0: 0.3383, 1: 0.8072, 2: 0.8051, 3: 0.4140} 3
    {0: -0.2640, 1: 0.2275, 2: 0.6286, 3: -0.0532} 2
    {0: 0.9050, 1: 0.6443, 2: 0.1270, 3: 0.4520} 2
    {0: 0.1874, 1: 0.4348, 2: 0.9819, 3: -0.0459} 2

    """

    def __init__(
        self,
        seed_model: int or np.random.RandomState = None,
        seed_sample: int or np.random.RandomState = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
    ):
        super().__init__(
            n_features=n_features, n_classes=n_classes, n_outputs=1, task=base.MULTI_CLF
        )
        self.seed_sample = seed_sample
        self.seed_model = seed_model
        self.n_num_features = n_features
        self.n_centroids = n_centroids
        self.centroids = None
        self.centroid_weights = None
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._generate_centroids()
        rng_sample = check_random_state(self.seed_sample)

        while True:
            x, y = self._generate_sample(rng_sample)
            yield x, y

    def _generate_sample(self, rng_sample: np.random.RandomState):
        idx = random_index_based_on_weights(self.centroid_weights, rng_sample)
        current_centroid = self.centroids[idx]
        att_vals = dict()
        magnitude = 0.0
        for i in range(self.n_features):
            att_vals[i] = (rng_sample.rand() * 2.0) - 1.0
            magnitude += att_vals[i] * att_vals[i]
        magnitude = np.sqrt(magnitude)
        desired_mag = rng_sample.normal() * current_centroid.std_dev
        scale = desired_mag / magnitude
        x = {i: current_centroid.centre[i] + att_vals[i] * scale for i in range(self.n_features)}
        y = current_centroid.class_label
        return x, y

    def _generate_centroids(self):
        """Generates centroids

        Sequentially creates all the centroids, choosing at random a center,
        a label, a standard deviation and a weight.

        """
        rng_model = check_random_state(self.seed_model)
        self.centroids = []
        self.centroid_weights = []
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(rng_model.rand())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = rng_model.randint(self.n_classes)
            self.centroids[i].std_dev = rng_model.rand()
            self.centroid_weights.append(rng_model.rand())


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
        Model's seed to generate centroids
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    seed_sample
        Sample's seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
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
    >>> from river import synth
    >>>
    >>> dataset = synth.RandomRBFDrift(seed_model=42, seed_sample=42,
    ...                                n_classes=4, n_features=4, n_centroids=20,
    ...                                change_speed=0.87, n_drift_centroids=10)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 1.1965, 1: 0.5729, 2: 0.8607, 3: 0.5888} 0
    {0: 0.3383, 1: 0.8072, 2: 0.8051, 3: 0.4140} 3
    {0: 0.5362, 1: -0.2867, 2: 0.0962, 3: 0.8974} 2
    {0: 1.1875, 1: 1.0385, 2: 0.8323, 3: -0.0553} 2
    {0: 0.3256, 1: 0.9206, 2: 0.8595, 3: 0.5907} 2


    """

    def __init__(
        self,
        seed_model: int or np.random.RandomState = None,
        seed_sample: int or np.random.RandomState = None,
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
        rng_sample = check_random_state(self.seed_sample)

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
        rng_model = check_random_state(self.seed_model)
        self.centroid_speed = []

        for i in range(self.n_drift_centroids):
            rand_speed = np.zeros(self.n_features)
            norm_speed = 0.0

            for j in range(self.n_features):
                rand_speed[j] = rng_model.rand()
                norm_speed += rand_speed[j] * rand_speed[j]

            norm_speed = np.sqrt(norm_speed)

            for j in range(self.n_features):
                rand_speed[j] /= norm_speed

            self.centroid_speed.append(rand_speed)


class Centroid:
    """ Class that stores a centroid's attributes. """

    def __init__(self):
        self.centre = None
        self.class_label = None
        self.std_dev = None


def random_index_based_on_weights(weights: list, random_state: np.random.RandomState):
    """Generate a random index, based on index weights and a random number generator.

    Parameters
    ----------
    weights
        The weights of the centroid's indexes.

    random_state
        Random number generator instance.

    Returns
    -------
    int
        The generated index.

    """
    prob_sum = np.sum(weights)
    val = random_state.rand() * prob_sum
    index = 0
    sum_value = 0.0
    while (sum_value <= val) & (index < len(weights)):
        sum_value += weights[index]
        index += 1
    return index - 1
