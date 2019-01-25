import collections
import functools
import math
import re

from sklearn import feature_extraction

from .. import base


__all__ = ['CountVectorizer', 'TFIDFVectorizer']


def compose(*functions):
    """Return a callable that chains multiple functions.

    Example
    -------

    >>> f = lambda x: x + 1
    >>> g = lambda x: x * 2
    >>> h = lambda x: -x
    >>> compose(f, g, h)(10)
    -22

    """

    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, reversed(functions), lambda x: x)


class VectorizerMixin:

    def __init__(self, on: str, strip_accents=True, lowercase=True, preprocessor=None,
                 tokenizer=None):
        self.on = on
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor or self.build_preprocessor()
        self.tokenizer = tokenizer or self.build_tokenizer()

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization."""
        steps = []

        if self.strip_accents:
            steps.append(feature_extraction.text.strip_accents_unicode)

        if self.lowercase:
            steps.append(str.lower)

        return compose(*steps) if steps else lambda x: x

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens."""
        token_pattern = re.compile(r'(?u)\b\w\w+\b')
        return lambda text: token_pattern.findall(text)


class CountVectorizer(base.Transformer, VectorizerMixin):
    """
    Example
    -------

        #!python
        >>> import creme
        >>> corpus = [
        ...     'This is the first document.',
        ...     'This document is the second document.',
        ...     'And this is the third one.',
        ...     'Is this the first document?',
        ... ]
        >>> vectorizer = creme.feature_extraction.CountVectorizer(on='sentence')
        >>> for sentence in corpus:
        ...     print(vectorizer.fit_one({'sentence': sentence}))
        Counter({'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1})
        Counter({'document': 2, 'this': 1, 'is': 1, 'the': 1, 'second': 1})
        Counter({'and': 1, 'this': 1, 'is': 1, 'the': 1, 'third': 1, 'one': 1})
        Counter({'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1})

    """

    def fit_one(self, x, y=None):
        return self.transform_one(x)

    def transform_one(self, x):
        return collections.Counter(self.tokenizer(self.preprocessor(x[self.on])))


class TFIDFVectorizer(base.Transformer, VectorizerMixin):

    def __init__(self, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.counter = CountVectorizer(**kwargs)
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

        # Compute the term counts
        term_counts = self.counter.fit_one(x)

        # Increment the document counter
        self.n += 1

        for term in term_counts:
            self.dfs[term] += 1

        return self.compute_tfidfs(term_counts)

    def transform_one(self, x):
        term_counts = self.counter.transform_one(x)
        return self.compute_tfidfs(term_counts)
