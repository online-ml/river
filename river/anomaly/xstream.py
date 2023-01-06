import math
import random
from collections import Counter

from river import anomaly
from river.transform.projection.streamhash_projector import StreamhashProjector
from river.utils.math import dict_zeros, get_minmax, merge

random.seed(14)


class xStream(anomaly.base.AnomalyDetector):
    """The xStream model for row-streaming data :cite:`xstream`. It first projects the data via streamhash projection. It then fits half space chains by reference windowing. It scores the instances using the window fitted to the reference window.

    Parameters
    ----------
    num_components
        Number of components for streamhash projection (Default=100).
    n_chains
        Number of half-space chains (Default=100).
    depth
        Maximum depth for the chains (Default=25).
    window_size
        Size (and the sliding length) of the reference window (Default=25).

    """

    def __init__(self, num_components=100, n_chains=100, depth=25, window_size=25):
        self.streamhash = StreamhashProjector(num_components=num_components)
        deltamax = {}
        for i in range(num_components):
            deltamax[i] = 0.5
        deltamax = {
            feature: 1.0 if (value <= 0.0001) else value for feature, value in deltamax.items()
        }
        self.window_size = window_size
        self.hs_chains = _HSChains(deltamax=deltamax, n_chains=n_chains, depth=depth)

        self.step = 0
        self.cur_window = []
        self.ref_window = None

    def learn_one(self, x, y=None):
        """Fits the model to next instance.

        Parameters
        ----------
        X
            Instance to learn.
        y
            Ignored since the model is unsupervised (Default=None).

        """
        self.step += 1

        X = self.streamhash.learn_one(x)
        X = self.streamhash.transform_one(X)
        self.cur_window.append(X)
        self.hs_chains.fit(X)

        if self.step % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.hs_chains.set_deltamax(deltamax)
            self.hs_chains.next_window()

        return self

    def score_one(self, x):
        """Scores the anomalousness of the next instance.

        Parameters
        ----------
        X
            Instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        """

        X = self.streamhash.learn_one(x)
        X = self.streamhash.transform_one(X)
        score = self.hs_chains.score(X)

        return score

    def _compute_deltamax(self):
        dico_min_max = {}
        for i in range(len(self.ref_window)):
            temp = {
                (i, feature): self.ref_window[i][feature] for feature in self.ref_window[i].keys()
            }
            dico_min_max = merge(dico_min_max, temp)
        mn, mx = get_minmax(dico_min_max)
        deltamax = {key: (mx[key] - mn[key]) / 2.0 for key in mx.keys()}
        deltamax = {
            feature: 1.0 if (value <= 0.0001) else value for feature, value in deltamax.items()
        }
        return deltamax


class _Chain:
    def __init__(self, deltamax, depth):
        k = len(deltamax)

        self.depth = depth
        self.fs = [random.randint(0, k - 1) for d in range(depth)]
        self.cmsketches = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth

        self.deltamax = deltamax  # feature ranges
        self.shift = {}
        self.rand_arr = []
        for key in self.deltamax.keys():
            rnd = random.random()
            self.shift[key] = rnd * self.deltamax[key]
            self.rand_arr.append(rnd)

        self.is_first_window = True

    def fit(self, x):
        prebins = {}
        dict_zeros(prebins, len(x))
        depthcount = {}
        dict_zeros(depthcount, len(self.deltamax))

        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[f] = (x[f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.shift[f] / self.deltamax[f]

            if self.is_first_window:
                cmsketch = self.cmsketches[depth]
                for key in prebins:
                    prebins[key] = math.floor(prebins[key])
                l_index = tuple(prebins.values())

                if l_index not in cmsketch:
                    cmsketch[l_index] = 0
                cmsketch[l_index] += 1

                self.cmsketches[depth] = cmsketch

                self.cmsketches_cur[depth] = cmsketch

            else:
                cmsketch = self.cmsketches_cur[depth]

                for key in prebins:
                    prebins[key] = math.floor(prebins[key])

                l_index = tuple(prebins.values())
                if l_index not in cmsketch:
                    cmsketch[l_index] = 0
                cmsketch[l_index] += 1

                self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, x):
        scores = {}
        dict_zeros(scores, self.depth)
        prebins = {}
        dict_zeros(prebins, len(x))
        depthcount = {}
        dict_zeros(depthcount, len(self.deltamax))

        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[f] = (x[f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.shift[f] / self.deltamax[f]

            cmsketch = self.cmsketches[depth]

            for key in prebins:
                prebins[key] = math.floor(prebins[key])

            l_index = tuple(prebins.values())
            if l_index not in cmsketch:
                scores[depth] = 0.0
            else:
                scores[depth] = cmsketch[l_index]
        return scores

    def score(self, x):
        scores = self.bincount(x)
        depths = {}
        for d in range(1, self.depth + 1):
            depths[d - 1] = d
        d_1 = dict(map(lambda x: (x[0], math.log2(x[1] + 1)), scores.items()))
        d_2 = dict(Counter(d_1) + Counter(depths))
        mini = -min(list(d_2.values()))
        return [mini]

    def next_window(self):
        self.is_first_window = False
        self.cmsketches = self.cmsketches_cur
        self.cmsketches_cur = [{} for _ in range(self.depth)] * self.depth


class _HSChains:
    """The Half-Space Chains approximates the density by computing neighborhood-counts at multiple scales.

    Parameters
    ----------
    deltamax
    n_chains
        Number of half-space chains (Default=100).
    depth
        Maximum depth for the chains (Default=25).

    """

    def __init__(self, deltamax, n_chains=100, depth=25):
        self.nchains = n_chains
        self.depth = depth
        self.chains = []

        for i in range(self.nchains):

            c = _Chain(deltamax=deltamax, depth=self.depth)
            self.chains.append(c)

    def score(self, x):
        scores = 0
        for ch in self.chains:
            scores += ch.score(x)[0]
        scores /= float(self.nchains)
        return [scores]

    def fit(self, x):
        for ch in self.chains:
            ch.fit(x)

    def next_window(self):
        for ch in self.chains:
            ch.next_window()

    def set_deltamax(self, deltamax):
        for ch in self.chains:
            ch.deltamax = deltamax
            for i in range(len(deltamax)):
                ch.shift[i] = ch.rand_arr[i] * deltamax[i]
