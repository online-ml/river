from __future__ import annotations

import collections
import functools
import operator

from sklearn.cluster import KMeans

from river import base, utils

from .base import DistanceFunc, FunctionWrapper


class SAMkNNClassifier(base.Classifier):
    """Self Adjusting Memory k-Nearest Neighbors (SAMkNN) for classification.

    High level description.

    Parameters
    ----------
    n_neighbors.
        Number of neighbors to use for the underlying k nearest neighbor
        classifier.
    max_mem_size.
        Maximum size of the Short and Long Term Memory combined.
    max_ltm_size.
        Maximum size of the Long Term Memory. If LTM reaches this size, it is
        compressed.
    min_stm_size.
        Minimum size of the Short Term Memory. Smaller sizes will not be
        considered while calculating optimal STM size.
    weighted.
        Use distance weighted kNN. If turned off majority voting is used.
    softmax.
        Apply softmax on the output probabilities.
    dist_func.
        Distance function to use for the k nearest neighbor classifier.
    recalculate_stm_error.
        Disables a heuristic that incrementally computes the interleaved-test-
        then-train accuracy for the optimal STM size estimation. Activating this
        increases runtime but may result in slightly better model performance.

    Notes
    -----
    As the LTM compression mechanism uses kmeans, SAM-kNN only works with
    nummerical features and every datapoint is required to have a value for
    every feature.

    Examples
    --------
    >>> from river import evaluate, metrics
    >>> from river.datasets import Bananas
    >>> from river.neighbors import SAMkNNClassifier

    >>> samknn = SAMkNNClassifier()
    >>> dataset = Bananas()

    >>> evaluate.progressive_val_score(dataset, samknn, metrics.Accuracy())
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        max_mem_size: int = 100,
        max_ltm_size: int = 50,
        min_stm_size: int = 10,
        weighted: bool = True,
        softmax: bool = False,
        dist_func: DistanceFunc | FunctionWrapper | None = None,
        recalculate_stm_error: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.max_mem_size = max_mem_size
        self.max_ltm_size = max_ltm_size
        self.min_stm_size = min_stm_size
        self.weighted = weighted
        self.softmax = softmax

        self.classes: set[base.typing.ClfTarget] = set()
        self.weights: dict[str, int] = {"stm": 0, "ltm": 0, "cm": 0}

        if dist_func is None:
            dist_func = functools.partial(utils.math.minkowski_distance, p=2)
        if not isinstance(dist_func, FunctionWrapper):
            dist_func = FunctionWrapper(dist_func)

        self.stm = SAMkNNShortTermMemory(
            n_neighbors=self.n_neighbors,
            dist_func=dist_func,
            min_stm_size=self.min_stm_size,
            weighted=self.weighted,
            recalculate_stm_error=recalculate_stm_error,
        )
        self.ltm = SAMkNNLongTermMemory(self.n_neighbors, dist_func=dist_func)

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y, **kwargs):
        self.classes.add(y)

        # Update memory weights
        for memory in self.weights.keys():
            self.weights[memory] += self.predict_one(x, memory=memory) == y

        # Append (x, y) to STM
        self.stm.append((x, y))

        # Check if max memory size is exceeded
        if self.stm.size() + self.ltm.size() > self.max_mem_size:
            # Transfer items from STM to LTM and compress LTM
            n_items_to_transfer = self.max_ltm_size - self.ltm.size()
            for item in self.stm.pop_n(n_items_to_transfer):
                self.ltm.append(item)
            self.ltm.compress()

        # Clean LTM with (x, y)
        clean_dist = self.stm.get_clean_distance((x, y))
        self.ltm.clean((x, y), clean_dist)

        # Determine optimal STM size
        optimal_stm_size = self.stm.optimial_size()
        if optimal_stm_size != self.stm.size():
            # Transfer items to LTM to achieve optimal STM size
            n_items_to_transfer = self.stm.size() - optimal_stm_size
            new_ltm_items = []
            for item in self.stm.pop_n(n_items_to_transfer):
                new_ltm_items.append(item)

            # Clean new LTM samples before appending
            cleaned_new_ltm_items = [
                new_ltm_item
                for new_ltm_item in new_ltm_items
                if all(
                    [
                        clean_dist == 0
                        or new_ltm_item[1] != stm_item[1]
                        or self.ltm.dist_func(new_ltm_item, stm_item) > clean_dist
                        for stm_item, clean_dist in zip(
                            self.stm, map(self.stm.get_clean_distance, self.stm)
                        )
                    ]
                )
            ]

            self.ltm.append(cleaned_new_ltm_items)

    def predict_proba_one(self, x, memory=None, **kwargs):
        # Select memory by weight, if none is specified
        if memory is None:
            memory = max(self.weights, key=self.weights.get)

        # Make predictions using the selected memory
        if memory == "stm":
            nearest = self.stm.search((x, None))
        elif memory == "ltm":
            nearest = self.ltm.search((x, None))
        else:
            nearest_stm = self.stm.search((x, None))
            nearest_ltm = self.ltm.search((x, None))
            nearest = sorted(nearest_stm + nearest_ltm, key=operator.itemgetter(1))[
                : self.n_neighbors
            ]

        # Create probability for each known class
        probas = {c: 0.0 for c in self.classes}

        # If no neighbors are found, return a uniform distribution
        if not nearest:
            return {cls: 1 / len(self.classes) for cls in self.classes}

        # Add up unnormalized probas
        for item, dist in nearest:
            probas[item[1]] += 1 / dist if self.weighted else 1

        # If softmax is enabled, return softmax probas
        if self.softmax:
            return utils.math.softmax(probas)

        # Return normalized probas
        return {cls: proba / sum(probas.values()) for cls, proba in probas.items()}


class SAMkNNMemory:
    def __init__(self, n_neighbors: int, dist_func: FunctionWrapper):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

        self.items: list[tuple[dict, base.typing.ClfTarget]] = []
        self.last_search_item: tuple[dict, base.typing.ClfTarget] = None

    def append(
        self, item: list[tuple[dict, base.typing.ClfTarget]] | tuple[dict, base.typing.ClfTarget]
    ):
        if isinstance(item, list):
            self.items += item
        else:
            self.items.append(item)

        self.last_search_item = None

    def size(self):
        return len(self.items)

    def search(self, item: tuple[dict, base.typing.ClfTarget], n_neighbors: int | None = None):
        # If search result is cached, return it
        if self.last_search_item is not None and self.last_search_item == item:
            return self.last_search_result

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Find nearest neighbors
        items_distances = ((p, self.dist_func(item, p)) for p in self.items)
        search_result = sorted(items_distances, key=operator.itemgetter(1))[:n_neighbors]

        # Cache this search result
        self.last_search_item = item
        self.last_search_result = search_result

        return search_result


class SAMkNNShortTermMemory(SAMkNNMemory):
    def __init__(
        self,
        n_neighbors: int,
        dist_func: FunctionWrapper,
        min_stm_size: int,
        weighted: bool,
        recalculate_stm_error: bool,
    ):
        super().__init__(n_neighbors, dist_func)
        self.min_stm_size = min_stm_size
        self.weighted = weighted
        self.recalculate_stm_error = recalculate_stm_error

        self.prediction_histories: list[bool] = {}

    def pop_n(self, n: int):
        # Invalidate cache and prediction histories as items are changed
        self.last_search_item = None
        self.prediction_histories = {}

        for _ in range(n):
            yield self.items.pop(0)

    def __iter__(self):
        yield from self.items

    def get_clean_distance(self, item: tuple[dict, base.typing.ClfTarget]):
        # As item itself is included in window,
        # search for self.n_neighbors+1 neighbors
        nearest = self.search(item, n_neighbors=self.n_neighbors + 1)
        furthest_distance_same_label = max(
            [item_dist[1] for item_dist in nearest if item_dist[0][1] == item[1]]
        )

        return furthest_distance_same_label

    def partial_interleaved_test_train_error(self, size: int):
        start_idx = len(self.items) - size

        if start_idx in self.prediction_histories.keys():
            # Make new prediction and append to prediction history
            item = self.items[-1]
            items_distances = ((p, self.dist_func(item, p)) for p in self.items[start_idx:-1])
            nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]

            probas = collections.defaultdict(lambda: 0)
            for item, dist in nearest:
                probas[item[1]] += 1 / dist if self.weighted else 1
            prediction = max(probas, key=probas.get)
            self.prediction_histories[start_idx].append(prediction == item[1])

        elif start_idx - 1 in self.prediction_histories.keys() and not self.recalculate_stm_error:
            # Use prediction history with start shifted by 1
            self.prediction_histories[start_idx] = self.prediction_histories[start_idx - 1]
            del self.prediction_histories[start_idx - 1]
            self.prediction_histories[start_idx].pop(0)

            # Make new prediction and append to prediction history
            item = self.items[-1]
            items_distances = ((p, self.dist_func(item, p)) for p in self.items[start_idx:-1])
            nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]

            probas = collections.defaultdict(lambda: 0)
            for item, dist in nearest:
                probas[item[1]] += 1 / dist if self.weighted else 1
            prediction = max(probas, key=probas.get)
            self.prediction_histories[start_idx].append(prediction == item[1])

        else:
            # Generate new Prediction history from scratch
            self.prediction_histories[start_idx] = []
            for cur_idx in range(start_idx + 1, len(self.items)):
                item = self.items[cur_idx]
                items_distances = (
                    (p, self.dist_func(item, p)) for p in self.items[start_idx:cur_idx]
                )
                nearest = sorted(items_distances, key=operator.itemgetter(1))[: self.n_neighbors]
                probas = collections.defaultdict(lambda: 0)
                for item, dist in nearest:
                    probas[item[1]] += 1 / dist if self.weighted else 1

                prediction = max(probas, key=probas.get)
                self.prediction_histories[start_idx].append(prediction == item[1])

        # Return interleaved-test-then-train accuracy
        return sum(self.prediction_histories[start_idx]) / len(self.prediction_histories[start_idx])

    def optimial_size(self):
        # Generate candidate sizes using repeated halving
        candidate_sizes = []
        cur_candidate_size = len(self.items)
        while cur_candidate_size > self.min_stm_size:
            candidate_sizes.append(cur_candidate_size)
            cur_candidate_size //= 2

        # If no alternative candidate sizes exist, return the current size
        if len(candidate_sizes) <= 1:
            return self.size()

        # Score all candidate sizes
        candidate_sizes_scores = {
            size: self.partial_interleaved_test_train_error(size) for size in candidate_sizes
        }

        # Delete unused prediction histories if necessary
        if self.recalculate_stm_error:
            for start_idx in list(self.prediction_histories.keys()):
                if len(self.items) - start_idx not in candidate_sizes:
                    del self.prediction_histories[start_idx]

        best_size = max(candidate_sizes_scores, key=candidate_sizes_scores.get)
        return best_size


class SAMkNNLongTermMemory(SAMkNNMemory):
    def compress(self):
        # Invalidate search cache, as items are compressed
        self.last_search_item = None

        # Class-wise, generate compressed items using clustering
        compressed_items = []
        classes = collections.Counter(sample[1] for sample in self.items)
        for cls, cls_count in classes.items():
            # Convert dict to lists
            fields, values = zip(
                *[tuple(zip(*item[0].items())) for item in self.items if item[1] == cls]
            )

            # Ensure that all items have the same features
            fields = set(fields)
            assert (
                len(fields) == 1
            ), "Not all datapoints have the same fields. Can not compress LTM!"
            fields = fields.pop()

            # Generate and add compressed data
            kmeans = KMeans(n_clusters=max(1, cls_count // 2), random_state=0)
            kmeans.fit(values)
            compressed_items += [
                ({cur_field: cur_value for cur_field, cur_value in zip(fields, cur_values)}, cls)
                for cur_values in kmeans.cluster_centers_
            ]

        # Overwrite items with compressed items
        self.items = compressed_items

    def clean(self, item: tuple[dict, base.typing.ClfTarget], clean_dist: float):
        # If the clean distance is 0, nothing needs to be done
        if clean_dist == 0:
            return

        # Clean items
        self.items = [
            cur_item
            for cur_item in self.items
            if cur_item[1] != item[1] or self.dist_func(cur_item, item) > clean_dist
        ]

        # Invalidate search cache as the items are changed
        self.last_search_item = None
