import collections
import copy
import random

from river import base, linear_model

__all__ = ["OutputCodeClassifier"]


def l1_dist(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


class OutputCodeClassifier(base.WrapperMixin, base.Classifier):
    """Output-code multiclass strategy.

    This also referred to as "error-correcting output codes".

    This class allows to learn a multi-class classification problem with a binary classifier. Each
    class is converted to a code of 0s and 1s. The length of the code is called  the code size. A
    copy of the classifier made for code. The codes associated with the classes are stored in a
    code book.

    When a new sample arrives, the label's code is retrieved from the code book. Then, each
    classifier is trained on the relevant part of code, which is either a 0 or a 1.

    For predicting, each classifier outputs a probability. These are then compared to each code in
    the code book, and the label which is the "closest" is chosen as the most likely class.
    Closeness is determined in terms of Manhattan distance.

    One specificity of online learning is that we don't how many classes there are initially.
    Therefore, a random procedure generates random codes on the fly whenever a previously unseed
    label appears.

    Parameters
    ----------
    classifier
        A binary classifier, although a multi-class classifier will work too.
    code_size
        The code size, which dictates how many copies of the provided classifiers to train. Must be
        strictly positive.
    seed
        A random seed number that can be set for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multiclass
    >>> from river import preprocessing

    >>> dataset = datasets.ImageSegments()

    >>> scaler = preprocessing.StandardScaler()
    >>> ooc = OutputCodeClassifier(
    ...     classifier=linear_model.LogisticRegression(),
    ...     code_size=10,
    ...     seed=24
    ... )
    >>> model = scaler | ooc

    >>> metric = metrics.MacroF1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MacroF1: 0.797119

    References
    ----------
    [^1]: [Dietterich, T.G. and Bakiri, G., 1994. Solving multiclass learning problems via error-correcting output codes. Journal of artificial intelligence research, 2, pp.263-286.](https://arxiv.org/pdf/cs/9501101.pdf)
    [^2]: [Allwein, E.L., Schapire, R.E. and Singer, Y., 2000. Reducing multiclass to binary: A unifying approach for margin classifiers. Journal of machine learning research, 1(Dec), pp.113-141.](https://www.cs.princeton.edu/~schapire/talks/ecoc-icml10.pdf)

    """

    def __init__(self, classifier: base.Classifier, code_size: int, seed: int = None):
        self.classifier = classifier
        self.code_size = code_size
        self.seed = seed
        self._rng = random.Random(seed)

        self.classifiers = {i: copy.deepcopy(classifier) for i in range(code_size)}

        # We don't how many classes there are, therefore we can't generate the code book
        # from the start. Therefore, we define a random queue of integers. When a new class
        # appears, we get the next integer and convert it to a code.
        integers = list(range(2 ** code_size))
        self._rng.shuffle(integers)
        self._ints = iter(integers)
        self.code_book = collections.defaultdict(self._next_code)

    def _next_code(self):
        i = next(self._ints)
        b = bin(i)[2:]  # convert to a string of 0s and 1s
        b = b.zfill(self.code_size)  # ensure the code is of length code_size
        return tuple(int(c) for c in b)

    @property
    def _multiclass(self):
        return True

    @property
    def _wrapped_model(self):
        return self.classifier

    @classmethod
    def _unit_test_params(cls):
        return {"classifier": linear_model.LogisticRegression(), "code_size": 6}

    def learn_one(self, x, y):

        code = self.code_book[y]

        for i, c in enumerate(code):
            self.classifiers[i].learn_one(x, c)

        return self

    def predict_one(self, x):

        if not self.code_book:  # it's empty
            return None

        output = [None for _ in range(self.code_size)]

        for i, clf in self.classifiers.items():
            output[i] = clf.predict_proba_one(x).get(True, 0.0)

        return min(self.code_book, key=lambda c: l1_dist(self.code_book[c], output))
