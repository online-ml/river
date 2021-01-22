from river import base
from river.utils import dict2numpy
from collections import Counter, deque
from collections.abc import Hashable
import numpy as np

__all__ = [
    "MLKNN",
]

def jensen_shannon(p, q, base=None):
    p = p / np.linalg.norm(p, ord=1)
    q = q / np.linalg.norm(q, ord=1)
    m = (p+q) / 2
    rel_entr_p_m = np.log(p/m).sum(axis=0)
    rel_entr_q_m = np.log(q/m).sum(axis=0)
    js = (rel_entr_p_m + rel_entr_q_m) / 2
    if base is not None:
        js /= np.log(base)
    return js

class MLKNN(base.Classifier, base.MultiOutputMixin):
    """A multi-output model that selects labels based on neighbors in an
    induced, dense vector space using Bayesian inference. See original
    publication_.

    .. _publication: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/pr07.pdf
    
    Parameters
    ----------
    distance_metric
        One of the strings 'l2', 'cos', 'js', or a callable which takes two
        vectors and produces a measure of their distance. 'L2' will use
        Euclidean distance, 'cos' will use cosine distance, and 'js' will use
        the Jensen–Shannon divergence for discrete probability distributions.
    smoothing
        A parameter selected to prevent overfitting. Reasonable values tend to
        lie between 0 and 1.
    n_neighbors
        The maximum number of neighbors to be used in the selection of
        candidate labels.
    window_size
        The number of samples and labels stored will never exceed window_size.
        If a value of None is given, the search space will grow arbitrarily
        large, but also be unbounded. Useful to prevent search in the induced
        vector space from becoming intractable and for prioritization of the
        most recent instances encountered during prediction.
    """

    __metrics = {
        'l2': lambda u, v: np.linalg.norm(u-v),
        'cos': lambda u, v: u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v),
        'js': jensen_shannon
    }

    def __init__(self, distance_metric='l2', smoothing=1.0, n_neighbors=10, window_size=1000):
        self.n_neighbors = n_neighbors
        if not callable(distance_metric):
            space = distance_metric.lower()
            if space not in self.__metrics:
                error = ('''
                         unrecognized metric '{}' for induced vector space:
                         please provide a function with two vector arguments
                         for distance computations, or one of 'cos', 'l2', or
                         'js'
                         ''').format(space)
                raise ValueError(' '.join(error.split()))
            distance_metric = self.__metrics[space]
        self.distance_metric = distance_metric
        self.smoothing = smoothing
        self.window_size = window_size
        self.X = deque(maxlen=window_size)
        self.Y = deque(maxlen=window_size)
        self.positive_frequencies = Counter()
        self.negative_frequencies = Counter()
        self.dim = None

        self._priors = None

    def labels(self):
        positives = self.positive_frequencies
        negatives = self.negative_frequencies
        labels = set(positives.keys()) | set(negatives.keys())
        return labels

    def _compute_priors(self):
        positives = self.positive_frequencies
        negatives = self.negative_frequencies
        labels = self.labels()
        positive_priors = dict()
        negative_priors = dict()
        s = self.smoothing
        n = len(self.Y)
        for l in labels:
            k = positives[l]
            p = (s+k) / (s*2+n)
            positive_priors[l] = p
            negative_priors[l] = 1-p
        return positive_priors, negative_priors

    def _knn_query(self, x, k=None):
        if k is None:
            k = self.n_neighbors
        distance = self.distance_metric
        distances = np.array([distance(x, v) for v in self.X])
        idx_asc_distance = np.argsort(distances)
        neighbor_ids = idx_asc_distance[:k]
        return neighbor_ids, distances[idx_asc_distance][:k]

    def _compute_membership_counts(self, x):
        x = dict2numpy(x)
        Y = self.Y
        yis, yds = self._knn_query(x)
        deltas = dict()
        for l in self.labels():
            delta = 0
            for yi in yis:
                y = Y[yi]
                y_has_label = l in y and y[l]
                if y_has_label:
                    delta += 1
            deltas[l] = delta
        return deltas

    def _compute_posteriors(self, x, deltas=None):
        if deltas is None:
            deltas = self._compute_membership_counts(x)
        x = dict2numpy(x)
        x.resize((self.dim,))
        Y = self.Y
        s = self.smoothing
        k = self.n_neighbors
        negative_posteriors = dict()
        positive_posteriors = dict()
        neighbor_ids, neighbor_distances = self._knn_query(x)
        neighbor_distances += neighbor_distances.min()
        neighbor_distances /= np.linalg.norm(neighbor_distances)
        for l, ci in deltas.items():
            c0 = np.zeros(k+1)
            c1 = np.zeros(k+1)
            for yi, yd in zip(neighbor_ids, neighbor_distances):
                y = Y[yi]
                c = c1 if l in y and y[l] else c0
                # This line deviates from the paper, where the count vector is
                # always incremented by one. instead of being a true "count,"
                # this implementation adds 2 for very similar entries, and
                # adds 1/2 for less similar entries.
                c[ci] += 2 * (1-yd)
            n0 = s*k + np.sum(c0)
            n1 = s*k + np.sum(c1)
            p0 = (s+c0) / n0
            p1 = (s+c1) / n1
            positive_posteriors[l] = p1
            negative_posteriors[l] = p0
        return positive_posteriors, negative_posteriors

    def learn_one(self, x, y):
        if isinstance(y, Hashable):
            y = {y: True}
        if self.dim is None:
            self.dim = len(x)
        x = dict2numpy(x)
        x.resize((self.dim,))
        assert x is not None
        positives = self.positive_frequencies
        negatives = self.negative_frequencies
        for t, p in y.items():
            counter = positives if p else negatives
            counter[t] += 1
        self.X.append(x)
        self.Y.append(y)

        self._priors = None

    def predict_proba_one(self, x):
        # memoize and retrieve deltas, priors, and posteriors if no training
        # has occurred since last prediction
        deltas = self._compute_membership_counts(x)
        pe1s, pe0s = self._compute_posteriors(x, deltas)
        if self._priors is None:
            ph1s, ph0s = self._compute_priors()
            self._priors = ph1s, ph0s
        else:
            ph1s, ph0s = self._priors
        conclusions = dict()
        for l, ci in deltas.items():
            pe1 = pe1s[l][ci]
            pe0 = pe0s[l][ci]
            ph1 = ph1s[l]
            ph0 = ph0s[l]
            p0 = pe0 * ph0
            p1 = pe1 * ph1
            conclusions[l] = {True: p1, False: p0}
        return conclusions

    def predict_one(self, x):
        posteriors = self.predict_proba_one(x)
        return {l: p[True] > p[False] for l, p in posteriors.items()}

    def reset(self):
        self.__init__(
            distance_metric=self.distance_metric,
            smoothing=self.smoothing,
            n_neighbors=self.n_neighbors,
            window_size=self.window_size
        )

def crossvalidate():
    from sklearn import datasets
    from tqdm import tqdm
    from river.multioutput import ClassifierChain
    from river.linear_model import LogisticRegression
    from river.stream import iter_sklearn_dataset
    from river.metrics import HammingLoss

    print('retrieve dataset...')
    samples = datasets.fetch_openml('yeast', version=4)
    print('initialize candidates...')
    logreg = LogisticRegression()
    baseline = ClassifierChain(model=logreg, order=list(range(14)))
    mlknn = MLKNN(distance_metric='cos', window_size=len(samples))
    dataset = iter_sklearn_dataset(dataset=samples, shuffle=True, seed=42)
    mlknn_performance = HammingLoss()
    baseline_performance = HammingLoss()
    print('evaluate ML-KNN vs baseline...')
    for x, y in tqdm(dataset, total=len(samples.data)):
        y = {i: p == 'TRUE' for i, (t, p) in enumerate(y.items())}
        baseline.learn_one(x, y)
        # evaluate/train baseline
        y_hat_baseline = baseline.predict_one(x)
        baseline_performance.update(y, y_hat_baseline)
        # evaluate/train ML-KNN
        mlknn.learn_one(x, y)
        y_hat_mlknn = mlknn.predict_one(x)
        y_hat_mlknn.update({t: False for t in y.keys() if t not in y_hat_mlknn})
        mlknn_performance.update(y, y_hat_mlknn)

    print(f'ClassifierChain: {baseline_performance}')
    print(f'MLKNN: {mlknn_performance}')

if __name__ == '__main__':
    crossvalidate()


