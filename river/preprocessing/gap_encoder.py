from __future__ import annotations

import collections

import numpy as np

from river import base
from river.feature_extraction.vectorize import strip_accents_unicode

__all__ = ["GapEncoder"]


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> collections.Counter:
    """Counts of contiguous character n-grams (spaces included).

    Like sklearn's `CountVectorizer(analyzer='char')`, except whitespace runs are not collapsed.

    """
    counts: collections.Counter = collections.Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(text) - n + 1):
            counts[text[i : i + n]] += 1
    return counts


class GapEncoder(base.Transformer):
    """Online Gamma-Poisson encoder for fuzzy string categories.

    This is an online version of [skrub's `GapEncoder`](https://skrub-data.org/stable/reference/generated/skrub.GapEncoder.html),
    learning one string at a time instead of in batches. Each string is turned into its character
    n-gram counts `v`, and we look for a small set of latent topics that explain those counts:
    `v ≈ h @ W`, where `W` holds the topic/n-gram weights and `h` says how strongly each topic is
    activated by the string. The counts are treated as Poisson with a Gamma prior on `h`, and both
    `h` and `W` are refined with multiplicative updates as data comes in.

    The point is fuzzy matching. Strings that mean the same thing usually share most of their
    n-grams ("london", "London, UK", "Lomdon"), so they light up the same topics even when they
    have no exact token in common. That makes the encoder handy for messy categorical text:
    hand-typed city names, job titles, chat messages full of typos, and so on.

    The n-gram vocabulary is not fixed ahead of time; it grows as new strings show up, the same
    way `preprocessing.LDA` grows its word vocabulary. Topics are kept up to date through two
    accumulators `A` and `B`, with a `half_life` (in samples) that lets old observations decay.
    `transform_one` is read-only: it only looks at n-grams it has already seen and never changes
    the model.

    Parameters
    ----------
    n_components
        Number of latent topics.
    on
        The name of the feature that contains the text to encode. If `None`, then each
        `learn_one` and `transform_one` should treat `x` as a `str` and not as a `dict`.
    strip_accents
        Whether or not to strip accent characters.
    lowercase
        Whether or not to convert all characters to lowercase.
    ngram_range
        The lower and upper boundary of the range of character n-grams to be extracted. All
        values of n such that `min_n <= n <= max_n` will be used.
    gamma_shape_prior
        Shape parameter of the Gamma prior on the activations.
    gamma_scale_prior
        Scale parameter of the Gamma prior on the activations.
    half_life
        Forgetting horizon for the topics, in number of samples. A sample's influence on the
        topic accumulators halves every `half_life` observations, i.e. the per-sample decay is
        `0.5 ** (1 / half_life)`. Larger values keep a longer memory and let the topics build up
        stable global structure; smaller values adapt faster to drift. Use `float('inf')` to
        never forget. This replaces the batch implementation's `rho`, which is not meaningful
        when updating one sample at a time (`rho = 0.5 ** (1 / half_life)`).
    max_iter_e_step
        Number of multiplicative iterations used to fit the activations of a sample during
        `learn_one`. `transform_one` always iterates until convergence (up to 100 iterations).
    seed
        Random number seed used for reproducibility. New vocabulary columns of `W` are
        initialized with Gamma-distributed random draws.

    Attributes
    ----------
    vocab : dict
        Maps each seen n-gram to its column index.
    W : np.ndarray
        Topic/n-gram weights, shape `(n_components, len(vocab))`. Rows sum to 1.
    A : np.ndarray
        Numerator accumulator of the topic updates, same shape as `W`.
    B : np.ndarray
        Denominator accumulator of the topic updates, shape `(n_components, 1)`.

    Examples
    --------

    Say people type in city names by hand. The same city shows up spelled several different ways,
    with typos and extra bits, and no two spellings need to share a whole word. The encoder still
    groups the variants under the same topic:

    >>> from river import preprocessing

    >>> enc = preprocessing.GapEncoder(n_components=2, seed=42)

    >>> X = ["london", "London", "London, UK", "Lomdon", "paris", "Paris", "Paris, France", "pqris"]
    >>> for _ in range(10):
    ...     for x in X:
    ...         enc.learn_one(x)

    Each variant of a city activates the topic of that city, even with typos never seen during
    training:

    >>> for x in ["London, UK", "Lndon", "Paris, France", "Pariss"]:
    ...     topics = enc.transform_one(x)
    ...     print(x, max(topics, key=topics.get), {k: round(v, 3) for k, v in topics.items()})
    London, UK 1 {0: 0.053, 1: 12.047}
    Lndon 1 {0: 0.05, 1: 3.05}
    Paris, France 0 {0: 16.547, 1: 0.053}
    Pariss 0 {0: 4.549, 1: 0.051}

    References
    ----------
    [^1]: [Cerda, P. and Varoquaux, G., 2020. Encoding high-cardinality string categorical variables. IEEE Transactions on Knowledge and Data Engineering.](https://inria.hal.science/hal-02171256v4)
    [^2]: [skrub's GapEncoder](https://skrub-data.org/stable/reference/generated/skrub.GapEncoder.html)

    """

    def __init__(
        self,
        n_components: int = 10,
        on: str | None = None,
        strip_accents: bool = True,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (2, 4),
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
        half_life: float = 1000.0,
        max_iter_e_step: int = 10,
        seed: int | None = None,
    ):
        self.n_components = n_components
        self.on = on
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.gamma_shape_prior = gamma_shape_prior
        self.gamma_scale_prior = gamma_scale_prior
        self.half_life = half_life
        self.decay = 0.5 ** (1.0 / half_life)
        self.max_iter_e_step = max_iter_e_step
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.vocab: dict[str, int] = {}
        self.W = np.zeros((n_components, 0))
        self.A = np.zeros((n_components, 0))
        self.B = np.full((n_components, 1), 1e-10)

    def _more_tags(self):
        if self.on is None:
            return {base.tags.TEXT_INPUT}
        return {}

    def _preprocess(self, x) -> str:
        text = x[self.on] if self.on is not None else x
        if self.strip_accents:
            text = strip_accents_unicode(text)
        if self.lowercase:
            text = text.lower()
        return text

    def _rescale_w(self) -> None:
        """Rescale the rows of `W` to sum to 1, keeping `A` consistent."""
        s = self.W.sum(axis=1, keepdims=True)
        self.W /= s
        self.A /= s

    def _fit_h(self, v: np.ndarray, idx: np.ndarray, max_iter: int) -> np.ndarray:
        """Fit the activations of one sample by multiplicative updates.

        Only the columns `idx` of `W`, i.e. the n-grams present in the sample, take part in the
        computation.

        """
        h = np.full(self.n_components, max(1e-10, v.sum()) / self.n_components)
        WT1 = 1 + 1 / self.gamma_scale_prior
        W_ = self.W[:, idx]
        W_WT1_ = W_ / WT1
        const = (self.gamma_shape_prior - 1) / WT1
        squared_epsilon = 1e-3**2
        for _ in range(max_iter):
            aux = W_WT1_ @ (v / (h @ W_ + 1e-10))
            h_out = h * aux + const
            squared_norm = np.dot(h_out - h, h_out - h) / np.dot(h, h)
            h = h_out
            if squared_norm <= squared_epsilon:
                break
        return h

    def learn_one(self, x):
        counts = _char_ngrams(self._preprocess(x), self.ngram_range)
        if not counts:
            return

        # Grow the vocabulary, and thus W and A, with the sample's unseen n-grams.
        new = [g for g in counts if g not in self.vocab]
        if new:
            for g in new:
                self.vocab[g] = len(self.vocab)
            self.W = np.concatenate(
                [
                    self.W,
                    self.rng.gamma(
                        shape=self.gamma_shape_prior,
                        scale=self.gamma_scale_prior,
                        size=(self.n_components, len(new)),
                    ),
                ],
                axis=1,
            )
            self.A = np.concatenate([self.A, np.full((self.n_components, len(new)), 1e-10)], axis=1)
            self._rescale_w()

        idx = np.fromiter((self.vocab[g] for g in counts), dtype=np.intp)
        v = np.fromiter(counts.values(), dtype=float)

        h = self._fit_h(v, idx, max_iter=self.max_iter_e_step)

        # Topic update through the decayed accumulators.
        self.A *= self.decay
        self.B *= self.decay
        ratio = v / (h @ self.W[:, idx] + 1e-10)
        self.A[:, idx] += self.W[:, idx] * np.outer(h, ratio)
        self.B += h[:, None]
        np.divide(self.A, self.B, out=self.W)
        self._rescale_w()

    def transform_one(self, x):
        counts = _char_ngrams(self._preprocess(x), self.ngram_range)
        known = {g: c for g, c in counts.items() if g in self.vocab}
        if not known:
            return {i: 0.0 for i in range(self.n_components)}

        idx = np.fromiter((self.vocab[g] for g in known), dtype=np.intp)
        v = np.fromiter(known.values(), dtype=float)
        h = self._fit_h(v, idx, max_iter=100)
        return dict(enumerate(h.tolist()))
