from skmultiflow.drift_detection import ADWIN
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.utils import check_random_state
from skmultiflow.trees.nodes import FoundNode
from skmultiflow.trees.nodes import LearningNodeNBAdaptive
from skmultiflow.trees.nodes import AdaNode

from skmultiflow.utils import get_max_value_key, normalize_values_in_dict

MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'
ERROR_WIDTH_THRESHOLD = 300


class AdaLearningNode(LearningNodeNBAdaptive, AdaNode):
    """ Learning node for Hoeffding Adaptive Tree that uses Adaptive Naive
    Bayes models.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_class_observations):
        super().__init__(initial_class_observations)
        self._estimation_error_weight = ADWIN()
        self.error_change = False
        self._randomSeed = 1
        self._classifier_random = check_random_state(self._randomSeed)

    # Override AdaNode
    def number_leaves(self):
        return 1

    # Override AdaNode
    def get_error_estimation(self):
        return self._estimation_error_weight.estimation

    # Override AdaNode
    def get_error_width(self):
        return self._estimation_error_weight.width

    # Override AdaNode
    def is_null_error(self):
        return self._estimation_error_weight is None

    def kill_tree_children(self, hat):
        pass

    # Override AdaNode
    def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
        true_class = y

        k = self._classifier_random.poisson(1.0)
        if k > 0:
            weight = weight * k

        class_prediction = get_max_value_key(self.get_class_votes(X, hat))

        bl_correct = (true_class == class_prediction)

        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to Adwin
        add = 0.0 if (bl_correct is True) else 1.0

        self._estimation_error_weight.add_element(add)
        # Detect change with Adwin
        self.error_change = self._estimation_error_weight.detected_change()

        if self.error_change is True and old_error > self.get_error_estimation():
            self.error_change = False

        # Update statistics
        super().learn_from_instance(X, y, weight, hat)

        # call ActiveLearningNode
        weight_seen = self.get_weight_seen()

        if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= hat.grace_period:
            hat._attempt_to_split(self, parent, parent_branch)
            self.set_weight_seen_at_last_split_evaluation(weight_seen)

    # Override LearningNodeNBAdaptive
    def get_class_votes(self, X, ht):
        # dist = {}
        prediction_option = ht.leaf_prediction
        # MC
        if prediction_option == MAJORITY_CLASS:
            dist = self.get_observed_class_distribution()
        # NB
        elif prediction_option == NAIVE_BAYES:
            dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
        # NBAdaptive (default)
        else:
            if self._mc_correct_weight > self._nb_correct_weight:
                dist = self.get_observed_class_distribution()
            else:
                dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

        dist_sum = sum(dist.values())  # sum all values in dictionary
        normalization_factor = dist_sum * self.get_error_estimation() * self.get_error_estimation()

        if normalization_factor > 0.0:
            dist = normalize_values_in_dict(dist, normalization_factor, inplace=False)

        return dist

    # Override AdaNode, New for option votes
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        found_nodes.append(FoundNode(self, parent, parent_branch))
