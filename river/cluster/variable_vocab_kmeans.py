import collections
import functools
import random
from typing import Dict, List

from river import base, utils

__all__ = ["VariableVocabKMeans"]


class VariableVocabKMeans(base.Clusterer):
    """Variable Vocabulary KMeans

    Instead of requiring a fixed set of features that are numerical, this version
    of Kmeans:

    1. Allows for providing a vocabulary (string variables) that are stored with the model
    2. Allows adding new words to the vocabulary.

    When we encounter a new word, given that it isn't in the vocabulary this means that
    we've never seen it before (and we know that the cluster centers can be given a  0
    value. The input parameters are the same s for KMeans.

    Parameters
    ----------
    n_clusters
        Maximum number of clusters to assign.
    halflife
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
    mu
        Mean of the normal distribution used to instantiate cluster positions.
    sigma
        Standard deviation of the normal distribution used to instantiate cluster positions.
    p
        Power parameter for the Minkowski metric. When `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.
    seed
        Random seed used for generating initial centroid positions.

    Attributes
    ----------
    vocab: dict
        Vocabulary that matches str tokens to their index in each center vector
    centers : dict
        Central positions of each cluster.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream

    # Instead of numbers, we provide vectors of tokens (or strings)
    >>> X = [
    ...    ["one", "two"],
    ...    ["one", "four"],
    ...    ["one", "zero"],
    ...    ["five", "six"],
    ...    ["seven", "eight"],
    ...    ["nine", "nine"]
    ]

    >>> model = cluster.VariableVocabKMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)

    >>> for i, vocab in enumerate(stream.iter_counts(X)):
    ...     model = model.learn_one(vocab)
    ...     center = model.predict_one(vocab)
    ...     print(f'{vocab} is assigned to cluster {center}')
    ...     # Get coord/value for each word
    ...     model.get_center_vocab(center)
    ...

    ... {'one': 1, 'two': 1} is assigned to cluster 0
    ... {'one': 1, 'four': 1} is assigned to cluster 0
    ... {'one': 1, 'zero': 1} is assigned to cluster 0
    ... {'five': 1, 'six': 1} is assigned to cluster 1
    ... {'seven': 1, 'eight': 1} is assigned to cluster 2
    ... {'nine': 2} is assigned to cluster 3
    """

    def __init__(
        self, n_clusters=5, halflife=0.5, mu=0, sigma=1, p=2, seed: int = None
    ):
        self.n_clusters = n_clusters
        self.halflife = halflife
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.seed = seed
        self._rng = random.Random(seed)
        rand_gauss = functools.partial(self._rng.gauss, self.mu, self.sigma)

        # Vocab is a lookup between vocab items and vector indices
        self.vocab = {}

        # Current index into vocab array
        self.index = 0
        self.centers = {
            i: collections.defaultdict(rand_gauss) for i in range(n_clusters)
        }  # type: ignore

    def get_center_vocab(self, center: int):
        """
        Given the id of a centroid, get the vocab and weights / counts for it.
        """
        # We need to be able to look up words based on index
        lookup = {idx: word for word, idx in self.vocab.items()}
        return {lookup[x]: count for x, count in self.centers[center].items()}

    def learn_predict_one(self, x: Dict[str, int]):
        """Equivalent to `k_means.learn_one(x).predict_one(x)`, but faster."""

        # Find the cluster with the closest center
        # Don't update vocab yet because it doesn't matter if we haven't
        # seen a token - it will return a count of 0.
        closest = self.predict_one(x)

        # Ensure centers have all features for future learning
        self.update_vocab(x)

        # Move the cluster's center (ONLY the one closest to!)
        # By this point all words are added to the vocabulary
        for word, count in x.items():
            xx = {self.vocab[word]: count}
            for i, xi in xx.items():
                self.centers[closest][i] += self.halflife * (
                    xi - self.centers[closest][i]
                )

        return closest

    def learn_one(self, x):
        self.learn_predict_one(x)
        return self

    def update_vocab(self, vocab: Dict[str, int]):
        """
        Given a vector of features, ensure we have each in our vocab
        """
        # We can do one dict update per new word
        updates = {}
        for word, count in vocab.items():
            if word not in self.vocab:
                self.vocab[word] = self.index

                # The word has never been seen by any previous centroid
                updates[self.index] = 0
                self.index += 1

        # This is akin to appending a new dimension to each vector
        for _, center in self.centers.items():
            center.update(updates)

    def predict_one(self, x: List[List[str]]):

        # Ensure we provide a lookup between features (vocab indices)
        # This should only include words we have seen before
        xx = {
            self.vocab.get(word): count
            for word, count in x.items()
            if word in self.vocab
        }

        def get_distance(c):
            return utils.math.minkowski_distance(a=self.centers[c], b=xx, p=self.p)

        return min(self.centers, key=get_distance)

    @classmethod
    def _unit_test_params(cls):
        yield {"n_clusters": 5}
