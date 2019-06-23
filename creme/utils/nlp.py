import functools
import re

from sklearn import feature_extraction


def compose(*functions):
    """Return a callable that chains multiple functions.

    Example:

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
    """Contains common processing steps used by each vectorizer.

    Parameters:
        on (str): The name of the feature that contains the text to vectorize. If `None`, then each
            ``fit_one`` and ``transform_one`` should treat ``x`` as a `str` and not as a ``dict``.
        strip_accents (bool): Whether or not to strip accent characters.
        lowercase (bool): Whether or not to convert all characters to lowercase.
        preprocessor (callable): The function used to preprocess the text. A default one is used
            if it is not provided by the user.
        tokenizer (callable): The function used to convert preprocessed text into a `dict` of
            tokens. A default one is used if it is not provided by the user.

    """

    def __init__(self, on=None, strip_accents=True, lowercase=True, preprocessor=None,
                 tokenizer=None):
        self.on = on
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocess = self.build_preprocessor() if preprocessor is None else preprocessor
        self.tokenize = re.compile(r'(?u)\b\w\w+\b').findall if tokenizer is None else tokenizer

    def _get_text(self, x):
        if self.on is not None:
            return x[self.on]
        return x

    def build_preprocessor(self):
        """Returns a function to preprocess the text before tokenization."""
        steps = []

        if self.strip_accents:
            steps.append(feature_extraction.text.strip_accents_unicode)

        if self.lowercase:
            steps.append(str.lower)

        return compose(*steps) if steps else lambda x: x
