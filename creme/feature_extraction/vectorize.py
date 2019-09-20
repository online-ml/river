import collections
import math
import re

from sklearn import feature_extraction

from .. import base


__all__ = ['CountVectorizer', 'TFIDFVectorizer']


class VectorizerMixin:
    """Contains common processing steps used by each vectorizer.

    Parameters:
        on (str): The name of the feature that contains the text to vectorize. If `None`, then each
            ``fit_one`` and ``transform_one`` should treat ``x`` as a `str` and not as a ``dict``.
        strip_accents (bool): Whether or not to strip accent characters.
        lowercase (bool): Whether or not to convert all characters to lowercase.
        tokenizer (callable): The function used to convert preprocessed text into a `dict` of
            tokens. A default one is used if it is not provided by the user.

    """

    def __init__(self, on=None, strip_accents=True, lowercase=True, tokenizer=None):
        self.on = on
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.tokenizer = re.compile(r'(?u)\b\w\w+\b').findall if tokenizer is None else tokenizer

    def _get_text(self, x):
        if self.on is not None:
            return x[self.on]
        return x

    def _more_tags(self):
        return {'handles_text': True}

    def preprocess(self, text):
        """Returns a function to preprocess the text before tokenization."""

        if self.strip_accents:
            text = feature_extraction.text.strip_accents_unicode(text)

        if self.lowercase:
            text = text.lower()

        return text


class CountVectorizer(base.Transformer, VectorizerMixin):
    """Counts the number of occurrences of each token.

    This returns exactly the same results as `sklearn.feature_extraction.text.CountVectorizer`.

    Parameters:
        on (str): The name of the feature that contains the text to vectorize. If ``None``, then
            the input is treated as a document instead of a set of features.
        strip_accents (bool): Whether or not to strip accent characters.
        lowercase (bool): Whether or not to convert all characters to lowercase.
        tokenizer (callable): The function used to convert preprocessed text into a `dict` of
            tokens. A default one is used if it is not provided by the user.

    Example:

        ::

            >>> import creme
            >>> corpus = [
            ...     'This is the first document.',
            ...     'This document is the second document.',
            ...     'And this is the third one.',
            ...     'Is this the first document?',
            ... ]
            >>> vectorizer = creme.feature_extraction.CountVectorizer()
            >>> for sentence in corpus:
            ...     print(vectorizer.transform_one(sentence))
            Counter({'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1})
            Counter({'document': 2, 'this': 1, 'is': 1, 'the': 1, 'second': 1})
            Counter({'and': 1, 'this': 1, 'is': 1, 'the': 1, 'third': 1, 'one': 1})
            Counter({'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1})

    """

    def transform_one(self, x):
        return collections.Counter(self.tokenizer(self.preprocess(self._get_text(x))))


class TFIDFVectorizer(base.Transformer, VectorizerMixin):
    """Computes values TF-IDF values.

    We use the same definition as scikit-learn. The only difference in the results comes the fact
    that the document frequencies have to be computed online.

    Parameters:
        on (str): The name of the feature that contains the text to vectorize. If ``None``, then
            the input is treated as a document instead of a set of features.
        strip_accents (bool): Whether or not to strip accent characters.
        lowercase (bool): Whether or not to convert all characters to lowercase.
        tokenizer (callable): The function used to convert preprocessed text into a `dict` of
            tokens. A default one is used if it is not provided by the user.
        normalize (bool): Whether or not the TF-IDF values by their L2 norm.

    Attributes:
        tfs (feature_extraction.CountVectorizer): The term counts.
        dfs (collections.defaultdict): The document counts.
        n (int): The number of scanned documents.

    Example:

        ::

            >>> import creme
            >>> corpus = [
            ...     'This is the first document.',
            ...     'This document is the second document.',
            ...     'And this is the third one.',
            ...     'Is this the first document?',
            ... ]
            >>> vectorizer = creme.feature_extraction.TFIDFVectorizer()
            >>> for sentence in corpus:
            ...     print(vectorizer.fit_one(sentence).transform_one(sentence))
            {'this': 0.447..., 'is': 0.447..., 'the': 0.447..., 'first': 0.447..., 'document': 0.447...}
            {'this': 0.333..., 'document': 0.667..., 'is': 0.333..., 'the': 0.333..., 'second': 0.469...}
            {'and': 0.497..., 'this': 0.293..., 'is': 0.293..., 'the': 0.293..., 'third': 0.497..., 'one': 0.497...}
            {'is': 0.384..., 'this': 0.384..., 'the': 0.384..., 'first': 0.580..., 'document': 0.469...}

    """

    def __init__(self, on=None, strip_accents=True, lowercase=True, tokenizer=None,
                 normalize=True):
        super().__init__(on, strip_accents, lowercase, tokenizer)
        self.normalize = normalize
        self.tfs = CountVectorizer(on, strip_accents, lowercase, tokenizer)
        self.dfs = collections.defaultdict(int)
        self.n = 0

    def compute_tfidfs(self, term_counts):
        n_terms = sum(term_counts.values())

        tfidfs = {}

        for term, count in term_counts.items():
            tf = count / n_terms
            idf = math.log(((1 + self.n) / (1 + self.dfs[term]))) + 1
            tfidfs[term] = tf * idf

        if self.normalize:
            norm = math.sqrt(sum(tfidf ** 2 for tfidf in tfidfs.values()))
            return {term: tfidf / norm for term, tfidf in tfidfs.items()}
        return tfidfs

    def fit_one(self, x, y=None):

        text = self._get_text(x)

        # Compute the term counts
        term_counts = self.tfs.fit_one(text).transform_one(text)

        # Increment the document frequencies of each term
        for term in term_counts:
            self.dfs[term] += 1

        # Increment the global document counter
        self.n += 1

        return self

    def transform_one(self, x):
        term_counts = self.tfs.transform_one(self._get_text(x))
        return self.compute_tfidfs(term_counts)
