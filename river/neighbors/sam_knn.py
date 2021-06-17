import copy as cp
import logging
from collections import deque

import numpy as np

from river.base import Classifier
from river.utils import dict2numpy

from . import libNearestNeighbor


class SAMKNNClassifier(Classifier):
    """Self Adjusting Memory coupled with the kNN classifier.

    The Self Adjusting Memory (SAM) [^1] model builds an ensemble with models targeting current
    or former concepts. SAM is built using two memories: STM for the current concept, and
    the LTM to retain information about past concepts. A cleaning process is in charge of
    controlling the size of the STM while keeping the information in the LTM consistent
    with the STM.

    Parameters
    ----------
    n_neighbors
        number of evaluated nearest neighbors.
    distance_weighting
        Type of weighting of the nearest neighbors. It `True`  will use 'distance'.
        Otherwise, will use 'uniform' (majority voting).
    window_size
         Maximum number of overall stored data points.
    ltm_size
        Proportion of the overall instances that may be used for the LTM. This is
        only relevant when the maximum number(maxSize) of stored instances is reached.
    stm_aprox_adaption
        Type of STM size adaption.<br/>
            - If `True` approximates the interleaved test-train error and is
               significantly faster than the exact version.<br/>
            - If `False` calculates the interleaved test-train error exactly for each of the
              evaluated window sizes, which often has to be recalculated from the scratch.<br/>
            - If `None`, the STM is not  adapted at all. If additionally `use_ltm=False`, then
              this algorithm is simply a kNN with fixed sliding window size.
    min_stm_size
        Minimum STM size which is evaluated during the STM size adaption.
    use_ltm
        Specifies whether the LTM should be used at all.

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors

    >>> dataset = synth.ConceptDriftStream(position=500, width=20, seed=1).take(1000)

    >>> model = neighbors.SAMKNNClassifier(window_size=100)

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)  # doctest: +SKIP
    Accuracy: 56.70%

    Notes
    -----
    This modules uses libNearestNeighbor, a C++ library used to speed up some of
    the algorithm's computations. When invoking the library's functions it's important
    to pass the right argument type. Although most of this framework's functionality
    will work with python standard types, the C++ library will work with 8-bit labels,
    which is already done by the SAMKNN class, but may be absent in custom classes that
    use SAMKNN static methods, or other custom functions that use the C++ library.

    References
    ----------
    [^1]: Losing, Viktor, Barbara Hammer, and Heiko Wersing.
          "Knn classifier with self adjusting memory for heterogeneous concept drift."
          In Data Mining (ICDM), 2016 IEEE 16th International Conference on,
          pp. 291-300. IEEE, 2016.

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        distance_weighting=True,
        window_size: int = 5000,
        ltm_size: float = 0.4,
        min_stm_size: int = 50,
        stm_aprox_adaption=True,
        use_ltm=True,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.distance_weighting = distance_weighting
        self.window_size = window_size
        self.ltm_size = ltm_size
        self.min_stm_size = min_stm_size
        self.use_ltm = use_ltm

        self._stm_samples = None
        self._stm_labels = np.empty(shape=0, dtype=np.int32)
        self._ltm_samples = None
        self._ltm_labels = np.empty(shape=0, dtype=np.int32)
        self.max_ltm_size = self.ltm_size * self.window_size
        self.max_stm_size = self.window_size - self.max_ltm_size
        self.min_stm_size = self.min_stm_size
        self.stm_aprox_adaption = stm_aprox_adaption

        self.stm_distances = np.zeros(shape=(window_size + 1, window_size + 1))
        if self.distance_weighting:
            self.get_labels_fct = SAMKNNClassifier._get_distance_weighted_label
        else:
            self.get_labels_fct = SAMKNNClassifier._get_maj_label
        if self.use_ltm:
            self.predict_fct = self._predict_by_all_memories
            self.size_check_fct = self._size_check_stmltm
        else:
            self.predict_fct = self._predict_by_stm
            self.size_check_fct = self._size_check_fade_out

        self.interleaved_pred_histories = {}
        self.ltm_pred_history = deque([])
        self.stm_pred_history = deque([])
        self.cmp_pred_history = deque([])

        self.train_step_count = 0
        self.stm_sizes = []
        self.ltm_sizes = []
        self.n_stm_correct = 0
        self.n_ltm_correct = 0
        self.n_cm_correct = 0
        self.n_possible_correct_predictions = 0
        self.n_correct_predictions = 0
        self.classifier_choice = []
        self.pred_history = []

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}

    @staticmethod
    def _get_distances(sample, samples):
        """Calculate distances from sample to all samples."""
        return libNearestNeighbor.get1ToNDistances(sample, samples)

    def _cluster_down(self, samples, labels):
        """
        Performs class-wise kMeans++ clustering for given samples with corresponding labels.
        The number of samples is halved per class.
        """
        from sklearn.cluster import KMeans

        logging.debug("cluster Down %d" % self.train_step_count)
        uniqueLabels = np.unique(labels)
        newSamples = np.empty(shape=(0, samples.shape[1]))
        newLabels = np.empty(shape=0, dtype=np.int32)
        for label in uniqueLabels:
            tmpSamples = samples[labels == label]
            newLength = int(max(tmpSamples.shape[0] / 2, 1))
            clustering = KMeans(n_clusters=newLength, n_init=1, random_state=0)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(
                newLabels, label * np.ones(shape=newLength, dtype=np.int32)
            )
        return newSamples, newLabels

    def _size_check_fade_out(self):
        """
        Makes sure that the STM does not surpass the maximum size.
        Only used when use_ltm=False.
        """
        STMShortened = False
        if len(self._stm_labels) > self.max_stm_size + self.max_ltm_size:
            STMShortened = True
            self._stm_samples = np.delete(self._stm_samples, 0, 0)
            self._stm_labels = np.delete(self._stm_labels, 0, 0)
            self.stm_distances[
                : len(self._stm_labels), : len(self._stm_labels)
            ] = self.stm_distances[
                1 : len(self._stm_labels) + 1, 1 : len(self._stm_labels) + 1
            ]

            if self.stm_aprox_adaption:
                key_set = list(self.interleaved_pred_histories.keys())
                # if self.interleaved_pred_histories.has_key(0):
                if 0 in key_set:
                    self.interleaved_pred_histories[0].pop(0)
                updated_histories = cp.deepcopy(self.interleaved_pred_histories)
                for key in self.interleaved_pred_histories.keys():
                    if key > 0:
                        if key == 1:
                            updated_histories.pop(0, None)
                        tmp = updated_histories[key]
                        updated_histories.pop(key, None)
                        updated_histories[key - 1] = tmp
                self.interleaved_pred_histories = updated_histories
            else:
                self.interleaved_pred_histories = {}
        return STMShortened

    def _size_check_stmltm(self):
        """
        Makes sure that the STM and LTM combined doe not surpass the maximum size.
        Only used when use_ltm=True.
        """
        stm_shortened = False
        if (
            len(self._stm_labels) + len(self._ltm_labels)
            > self.max_stm_size + self.max_ltm_size
        ):
            if len(self._ltm_labels) > self.max_ltm_size:
                self._ltm_samples, self._ltm_labels = self._cluster_down(
                    self._ltm_samples, self._ltm_labels
                )
            else:
                if (
                    len(self._stm_labels) + len(self._ltm_labels)
                    > self.max_stm_size + self.max_ltm_size
                ):
                    stm_shortened = True
                    n_shifts = int(self.max_ltm_size - len(self._ltm_labels) + 1)
                    shift_range = range(n_shifts)
                    self._ltm_samples = np.vstack(
                        [self._ltm_samples, self._stm_samples[:n_shifts, :]]
                    )
                    self._ltm_labels = np.append(
                        self._ltm_labels, self._stm_labels[:n_shifts]
                    )
                    self._ltm_samples, self._ltm_labels = self._cluster_down(
                        self._ltm_samples, self._ltm_labels
                    )
                    self._stm_samples = np.delete(self._stm_samples, shift_range, 0)
                    self._stm_labels = np.delete(self._stm_labels, shift_range, 0)
                    self.stm_distances[
                        : len(self._stm_labels), : len(self._stm_labels)
                    ] = self.stm_distances[
                        n_shifts : len(self._stm_labels) + n_shifts,
                        n_shifts : len(self._stm_labels) + n_shifts,
                    ]
                    for _ in shift_range:
                        self.ltm_pred_history.popleft()
                        self.stm_pred_history.popleft()
                        self.cmp_pred_history.popleft()
                    self.interleaved_pred_histories = {}
        return stm_shortened

    def _clean_samples(self, samples_cl, labels_cl, only_last=False):
        """
        Removes distance-based all instances from the input samples that
        contradict those in the STM.
        """
        if len(self._stm_labels) > self.n_neighbors and samples_cl.shape[0] > 0:
            if only_last:
                loop_range = [len(self._stm_labels) - 1]
            else:
                loop_range = range(len(self._stm_labels))
            for i in loop_range:
                if len(labels_cl) == 0:
                    break
                samples_shortened = np.delete(self._stm_samples, i, 0)
                labels_shortened = np.delete(self._stm_labels, i, 0)
                distances_stm = SAMKNNClassifier._get_distances(
                    self._stm_samples[i, :], samples_shortened
                )
                nn_indices_stm = libNearestNeighbor.nArgMin(
                    self.n_neighbors, distances_stm
                )[0]
                distances_ltm = SAMKNNClassifier._get_distances(
                    self._stm_samples[i, :], samples_cl
                )
                nn_indices_ltm = libNearestNeighbor.nArgMin(
                    min(len(distances_ltm), self.n_neighbors), distances_ltm
                )[0]
                correct_indices_stm = nn_indices_stm[
                    labels_shortened[nn_indices_stm] == self._stm_labels[i]
                ]
                if len(correct_indices_stm) > 0:
                    dist_threshold = np.max(distances_stm[correct_indices_stm])
                    wrong_indices_ltm = nn_indices_ltm[
                        labels_cl[nn_indices_ltm] != self._stm_labels[i]
                    ]
                    del_indices = np.where(
                        distances_ltm[wrong_indices_ltm] <= dist_threshold
                    )[0]
                    samples_cl = np.delete(
                        samples_cl, wrong_indices_ltm[del_indices], 0
                    )
                    labels_cl = np.delete(labels_cl, wrong_indices_ltm[del_indices], 0)
        return samples_cl, labels_cl

    def _learn_one(self, x, y):
        """Processes a new sample."""
        distances_stm = SAMKNNClassifier._get_distances(x, self._stm_samples)
        if not self.use_ltm:
            self._learn_one_by_stm(x, y, distances_stm)
        else:
            self._learn_one_by_all_memories(x, y, distances_stm)

        self.train_step_count += 1
        self._stm_samples = np.vstack([self._stm_samples, x])
        self._stm_labels = np.append(self._stm_labels, y)
        stm_shortened = self.size_check_fct()

        self._ltm_samples, self._ltm_labels = self._clean_samples(
            self._ltm_samples, self._ltm_labels, only_last=True
        )

        if self.stm_aprox_adaption is not None:
            if stm_shortened:
                distances_stm = SAMKNNClassifier._get_distances(
                    x, self._stm_samples[:-1, :]
                )

            self.stm_distances[
                len(self._stm_labels) - 1, : len(self._stm_labels) - 1
            ] = distances_stm
            old_window_size = len(self._stm_labels)
            (
                new_window_size,
                self.interleaved_pred_histories,
            ) = STMSizer.get_new_stm_size(
                self.stm_aprox_adaption,
                self._stm_labels,
                self.n_neighbors,
                self.get_labels_fct,
                self.interleaved_pred_histories,
                self.stm_distances,
                self.min_stm_size,
            )

            if new_window_size < old_window_size:
                del_range = range(old_window_size - new_window_size)
                old_stm_samples = self._stm_samples[del_range, :]
                old_stm_labels = self._stm_labels[del_range]
                self._stm_samples = np.delete(self._stm_samples, del_range, 0)
                self._stm_labels = np.delete(self._stm_labels, del_range, 0)
                self.stm_distances[
                    : len(self._stm_labels), : len(self._stm_labels)
                ] = self.stm_distances[
                    (old_window_size - new_window_size) : (
                        (old_window_size - new_window_size) + len(self._stm_labels)
                    ),
                    (old_window_size - new_window_size) : (
                        (old_window_size - new_window_size) + len(self._stm_labels)
                    ),
                ]

                if self.use_ltm:
                    for _ in del_range:
                        self.stm_pred_history.popleft()
                        self.ltm_pred_history.popleft()
                        self.cmp_pred_history.popleft()

                    old_stm_samples, old_stm_labels = self._clean_samples(
                        old_stm_samples, old_stm_labels
                    )
                    self._ltm_samples = np.vstack([self._ltm_samples, old_stm_samples])
                    self._ltm_labels = np.append(self._ltm_labels, old_stm_labels)
                    self.size_check_fct()
        self.stm_sizes.append(len(self._stm_labels))
        self.ltm_sizes.append(len(self._ltm_labels))

    def _learn_one_by_all_memories(self, sample, label, distances_stm):
        """
        Predicts the label of a given sample by using the STM, LTM and the CM.
        Only used when use_ltm=True.
        """
        predicted_label_ltm = 0
        predicted_label_stm = 0
        predicted_label_both = 0
        classifier_choice = 0
        if len(self._stm_labels) == 0:
            predicted_label = predicted_label_stm
        else:
            if len(self._stm_labels) < self.n_neighbors:
                predicted_label_stm = self.get_labels_fct(
                    distances_stm, self._stm_labels, len(self._stm_labels)
                )[0]
                predicted_label = predicted_label_stm
            else:
                distances_ltm = SAMKNNClassifier._get_distances(
                    sample, self._ltm_samples
                )
                predicted_label_stm = self.get_labels_fct(
                    distances_stm, self._stm_labels, self.n_neighbors
                )[0]
                predicted_label_both = self.get_labels_fct(
                    np.append(distances_stm, distances_ltm),
                    np.append(self._stm_labels, self._ltm_labels),
                    self.n_neighbors,
                )[0]

                if len(self._ltm_labels) >= self.n_neighbors:  # noqa
                    predicted_label_ltm = self.get_labels_fct(
                        distances_ltm, self._ltm_labels, self.n_neighbors
                    )[0]
                    correct_ltm = np.sum(self.ltm_pred_history)
                    correct_stm = np.sum(self.stm_pred_history)
                    correct_both = np.sum(self.cmp_pred_history)
                    labels = [
                        predicted_label_stm,
                        predicted_label_ltm,
                        predicted_label_both,
                    ]
                    classifier_choice = np.argmax(
                        [correct_stm, correct_ltm, correct_both]
                    )
                    predicted_label = labels[classifier_choice]  # noqa
                else:
                    predicted_label = predicted_label_stm

        self.classifier_choice.append(classifier_choice)
        self.cmp_pred_history.append(predicted_label_both == label)
        self.n_cm_correct += predicted_label_both == label
        self.stm_pred_history.append(predicted_label_stm == label)
        self.n_stm_correct += predicted_label_stm == label
        self.ltm_pred_history.append(predicted_label_ltm == label)
        self.n_ltm_correct += predicted_label_ltm == label
        self.n_possible_correct_predictions += label in [
            predicted_label_stm,
            predicted_label_both,
            predicted_label_ltm,
        ]
        self.n_correct_predictions += predicted_label == label
        return predicted_label

    def _predict_by_all_memories(self, sample, label, distances_stm):  # noqa
        predicted_label_stm = 0
        if len(self._stm_labels) == 0:
            predicted_label = predicted_label_stm
        else:
            if len(self._stm_labels) < self.n_neighbors:
                predicted_label_stm = self.get_labels_fct(
                    distances_stm, self._stm_labels, len(self._stm_labels)
                )[0]
                predicted_label = predicted_label_stm
            else:
                distances_ltm = SAMKNNClassifier._get_distances(
                    sample, self._ltm_samples
                )
                predicted_label_stm = self.get_labels_fct(
                    distances_stm, self._stm_labels, self.n_neighbors
                )[0]
                distances_new = cp.deepcopy(distances_stm)
                stm_labels_new = cp.deepcopy(self._stm_labels)
                predicted_label_both = self.get_labels_fct(
                    np.append(distances_new, distances_ltm),
                    np.append(stm_labels_new, self._ltm_labels),
                    self.n_neighbors,
                )[0]
                if len(self._ltm_labels) >= self.n_neighbors:  # noqa
                    predicted_label_ltm = self.get_labels_fct(
                        distances_ltm, self._ltm_labels, self.n_neighbors
                    )[0]
                    correct_ltm = np.sum(self.ltm_pred_history)
                    correct_stm = np.sum(self.stm_pred_history)
                    correct_both = np.sum(self.cmp_pred_history)
                    labels = [
                        predicted_label_stm,
                        predicted_label_ltm,
                        predicted_label_both,
                    ]
                    classifier_choice = np.argmax(
                        [correct_stm, correct_ltm, correct_both]
                    )
                    predicted_label = labels[classifier_choice]  # noqa
                else:
                    predicted_label = predicted_label_stm

        return predicted_label

    def _learn_one_by_stm(self, sample, label, distances_stm):
        pass

    def _predict_by_stm(self, sample, label, distances_stm):  # noqa
        """Predicts the label of a given sample by the STM, only used when use_ltm=False."""
        predicted_label = 0
        curr_len = len(self._stm_labels)
        if curr_len > 0:
            predicted_label = self.get_labels_fct(
                distances_stm, self._stm_labels, min(self.n_neighbors, curr_len)
            )[0]
        return predicted_label

    def learn_one(self, x, y) -> "Classifier":
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
        x
            The sample's features
        y
            The sample's class label.

        Returns
        -------
        self
        """
        x_array = dict2numpy(x)
        c = len(x_array)
        if self._stm_samples is None:
            self._stm_samples = np.empty(shape=(0, c))
            self._ltm_samples = np.empty(shape=(0, c))

        self._learn_one(x_array, y)

        return self

    def predict_one(self, x: dict):
        x_array = dict2numpy(x)
        c = len(x_array)
        if self._stm_samples is None:
            self._stm_samples = np.empty(shape=(0, c))
            self._ltm_samples = np.empty(shape=(0, c))

        distances_stm = SAMKNNClassifier._get_distances(x_array, self._stm_samples)
        return self.predict_fct(x_array, None, distances_stm)

    def predict_proba_one(self, x):
        raise NotImplementedError

    @staticmethod
    def _get_maj_label(distances, labels, n_neighbors):
        """Returns the majority label of the k nearest neighbors."""

        nn_indices = libNearestNeighbor.nArgMin(n_neighbors, distances)

        if not isinstance(labels, type(np.array([]))):
            labels = np.asarray(labels, dtype=np.int8)
        else:
            labels = np.int8(labels)

        pred_labels = libNearestNeighbor.mostCommon(labels[nn_indices])

        return pred_labels

    @staticmethod
    def _get_distance_weighted_label(distances, labels, n_neighbors):
        """Returns the the distance weighted label of the k nearest neighbors."""
        nn_indices = libNearestNeighbor.nArgMin(n_neighbors, distances)
        sqrtDistances = np.sqrt(distances[nn_indices])
        if not isinstance(labels, type(np.array([]))):
            labels = np.asarray(labels, dtype=np.int8)
        else:
            labels = np.int8(labels)

        predLabels = libNearestNeighbor.getLinearWeightedLabels(
            labels[nn_indices], sqrtDistances
        )
        return predLabels

    @property
    def STMSamples(self):  # noqa
        """Samples in the STM."""
        return self._stm_samples

    @property
    def STMLabels(self):  # noqa
        """Class labels in the STM."""
        return self._stm_labels

    @property
    def LTMSamples(self):  # noqa
        """Samples in the LTM."""
        return self._ltm_samples

    @property
    def LTMLabels(self):  # noqa
        """Class labels in the LTM."""
        return self._ltm_labels


class STMSizer:
    """Utility class to adapt the size of the sliding window of the STM."""

    @staticmethod
    def get_new_stm_size(
        aprox_adaption_strategy,
        labels,
        n_neighbours,
        get_labels_fct,
        prediction_histories,
        distances_stm,
        min_stm_size,
    ):
        """Returns the new STM size."""
        if aprox_adaption_strategy:
            "Use approximate interleaved test-train error"
            return STMSizer._get_max_acc_approx_window_size(
                labels,
                n_neighbours,
                get_labels_fct,
                prediction_histories,
                distances_stm,
                min_size=min_stm_size,
            )
        elif aprox_adaption_strategy is not None and not aprox_adaption_strategy:
            "Use exact interleaved test-train error"
            return STMSizer._get_max_acc_window_size(
                labels,
                n_neighbours,
                get_labels_fct,
                prediction_histories,
                distances_stm,
                min_size=min_stm_size,
            )
        elif aprox_adaption_strategy is None:
            "No stm adaption"
            return len(labels), prediction_histories
        else:
            raise Exception(f"Invalid adaption_strategy: {aprox_adaption_strategy}")

    @staticmethod
    def _acc_score(y_pred, y_true):
        """Calculates the achieved accuracy."""
        return np.sum(y_pred == y_true) / float(len(y_pred))

    @staticmethod
    def _get_interleaved_test_train_acc(
        labels, n_neighbours, get_labels_fct, distances_stm
    ):
        """Calculates the interleaved test train accuracy from the scratch."""
        predLabels = []
        for i in range(n_neighbours, len(labels)):
            distances = distances_stm[i, :i]
            predLabels.append(get_labels_fct(distances, labels[:i], n_neighbours)[0])
        return (
            STMSizer._acc_score(predLabels[:], labels[n_neighbours:]),
            (predLabels == labels[n_neighbours:]).tolist(),
        )

    @staticmethod
    def _get_interleaved_test_train_acc_pred_history(
        labels, n_neighbours, get_labels_fct, prediction_history, distances_stm
    ):
        """
        Calculates the interleaved test train accuracy incrementally
        by using the previous predictions.
        """
        for i in range(len(prediction_history) + n_neighbours, len(labels)):
            distances = distances_stm[i, :i]
            label = get_labels_fct(distances, labels[:i], n_neighbours)[0]
            prediction_history.append(label == labels[i])
        return (
            np.sum(prediction_history) / float(len(prediction_history)),
            prediction_history,
        )

    @staticmethod
    def _adapt_histories(n_deletions, prediction_histories):
        """
        Removes predictions of the largest window size and shifts
        the remaining ones accordingly.
        """
        for i in range(n_deletions):
            sortedKeys = np.sort(list(prediction_histories.keys()))
            prediction_histories.pop(sortedKeys[0], None)
            delta = sortedKeys[1]
            for j in range(1, len(sortedKeys)):
                prediction_histories[sortedKeys[j] - delta] = prediction_histories.pop(
                    sortedKeys[j]
                )
        return prediction_histories

    @staticmethod
    def _get_max_acc_window_size(
        labels,
        n_neighbours,
        get_labels_fct,
        prediction_histories,
        distances_stm,
        min_size=50,
    ):
        """
        Returns the window size with the minimum interleaved
        test-train error (exact calculation).
        """
        n_samples = len(labels)
        if n_samples < 2 * min_size:
            return n_samples, prediction_histories
        else:
            numSamplesRange = [n_samples]
            while numSamplesRange[-1] / 2 >= min_size:
                numSamplesRange.append(numSamplesRange[-1] / 2)

            accuracies = []
            keys_to_remove = []
            for key in prediction_histories.keys():
                if key not in (n_samples - np.array(numSamplesRange)):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                prediction_histories.pop(key, None)

            for numSamplesIt in numSamplesRange:
                idx = int(n_samples - numSamplesIt)
                keyset = list(prediction_histories.keys())
                # if predictionHistories.has_key(idx):
                if idx in keyset:
                    (
                        accuracy,
                        predHistory,
                    ) = STMSizer._get_interleaved_test_train_acc_pred_history(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        prediction_histories[idx],
                        distances_stm[idx:, idx:],
                    )
                else:
                    accuracy, predHistory = STMSizer._get_interleaved_test_train_acc(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        distances_stm[idx:, idx:],
                    )
                prediction_histories[idx] = predHistory
                accuracies.append(accuracy)
            accuracies = np.round(accuracies, decimals=4)
            best_n_train_idx = np.argmax(accuracies)
            window_size = numSamplesRange[best_n_train_idx]  # noqa

            if window_size < n_samples:
                prediction_histories = STMSizer._adapt_histories(
                    best_n_train_idx, prediction_histories
                )
            return int(window_size), prediction_histories

    @staticmethod
    def _get_max_acc_approx_window_size(
        labels,
        n_neighbours,
        get_labels_fct,
        prediction_histories,
        distances_stm,
        min_size=50,
    ):
        """
        Returns the window size with the minimum interleaved
        test-train error (using an approximation).
        """
        n_samples = len(labels)
        if n_samples < 2 * min_size:
            return n_samples, prediction_histories
        else:
            n_samples_range = [n_samples]
            while n_samples_range[-1] / 2 >= min_size:
                n_samples_range.append(n_samples_range[-1] / 2)
            accuracies = []
            for numSamplesIt in n_samples_range:
                idx = int(n_samples - numSamplesIt)
                keyset = list(prediction_histories.keys())
                # if predictionHistories.has_key(idx):
                if idx in keyset:
                    (
                        accuracy,
                        predHistory,
                    ) = STMSizer._get_interleaved_test_train_acc_pred_history(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        prediction_histories[idx],
                        distances_stm[idx:, idx:],
                    )
                # elif predictionHistories.has_key(idx-1):
                elif idx - 1 in keyset:
                    predHistory = prediction_histories[idx - 1]
                    prediction_histories.pop(idx - 1, None)
                    predHistory.pop(0)
                    (
                        accuracy,
                        predHistory,
                    ) = STMSizer._get_interleaved_test_train_acc_pred_history(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        predHistory,
                        distances_stm[idx:, idx:],
                    )
                else:
                    accuracy, predHistory = STMSizer._get_interleaved_test_train_acc(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        distances_stm[idx:, idx:],
                    )
                prediction_histories[idx] = predHistory
                accuracies.append(accuracy)
            accuracies = np.round(accuracies, decimals=4)
            best_n_train_idx = np.argmax(accuracies)
            if best_n_train_idx > 0:
                moreAccurateIndices = np.where(accuracies > accuracies[0])[0]
                for i in moreAccurateIndices:
                    idx = int(n_samples - n_samples_range[i])
                    accuracy, predHistory = STMSizer._get_interleaved_test_train_acc(
                        labels[idx:],
                        n_neighbours,
                        get_labels_fct,
                        distances_stm[idx:, idx:],
                    )
                    prediction_histories[idx] = predHistory
                    accuracies[i] = accuracy
                accuracies = np.round(accuracies, decimals=4)
                best_n_train_idx = np.argmax(accuracies)
            window_size = n_samples_range[best_n_train_idx]  # noqa

            if window_size < n_samples:
                prediction_histories = STMSizer._adapt_histories(
                    best_n_train_idx, prediction_histories
                )
            return int(window_size), prediction_histories
