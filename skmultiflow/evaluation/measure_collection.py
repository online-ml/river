__author__ = 'Guilherme Matsumoto'

import skmultiflow.evaluation.metrics.metrics as metrics
import numpy as np
from skmultiflow.core.base_object import BaseObject
from skmultiflow.core.utils.data_structures import FastBuffer, FastComplexBuffer, ConfusionMatrix, MOLConfusionMatrix


class ClassificationMeasurements(BaseObject):
    """
        i -> true labels
        j -> predictions
    """
    def __init__(self, targets=None, dtype=np.int64):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None
        self.sample_count = 0
        self.majority_classifier = 0
        self.correct_no_change = 0
        self.targets = targets

    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.sample_count = 0
        self.majority_classifier = 0
        self.correct_no_change = 0
        self.confusion_matrix.restart(self.n_targets)
        pass

    def add_result(self, sample, prediction):
        true_y = self._get_target_index(sample, True)
        pred = self._get_target_index(prediction, True)
        self.confusion_matrix.update(true_y, pred)
        self.sample_count += 1

        self.majority_classifier = (self.majority_classifier + 1) if (self.get_majority_class() == sample) else self.majority_classifier
        self.correct_no_change = (self.correct_no_change + 1) if (self.last_true_label == sample) else self.correct_no_change

        self.last_true_label = sample
        self.last_prediction = prediction

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_majority_class(self):
        """ Get the true majority class
        
        :return: 
        """
        if (self.n_targets is None) or (self.n_targets == 0):
           return False
        majority_class = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum = 0.0
            for j in range(self.n_targets):
                sum += self.confusion_matrix.value_at(i, j)
            sum = sum / self.sample_count
            if sum > max_prob:
                max_prob = sum
                majority_class = i

        return majority_class

    def get_performance(self):
        sum = 0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            sum += self.confusion_matrix.value_at(i, i)
        return sum / self.sample_count

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_performance()

    def _get_target_index(self, target, add = False):
        if (self.targets is None) and add:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        elif (self.targets is None) and (not add):
            return None
        if ((target not in self.targets) and (add)):
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        p0 = self.get_performance()
        pc = 0.0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.sample_count
            sum_column = np.sum(column) / self.sample_count

            pc += sum_row * sum_column
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_t(self):
        p0 = self.get_performance()
        if self.sample_count != 0:
            pc = self.correct_no_change / self.sample_count
        else:
            pc =0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_m(self):
        p0 = self.get_performance()
        if self.sample_count != 0:
            pc = self.majority_classifier / self.sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'


class WindowClassificationMeasurements(BaseObject):
    def __init__(self, targets=None, dtype=np.int64, window_size=200):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_class = None

        self.targets = targets
        self.window_size = window_size
        self.true_labels = FastBuffer(window_size)
        self.predictions = FastBuffer(window_size)
        self.temp = 0
        self.last_prediction = None
        self.last_true_label = None

        self.majority_classifier = 0
        self.correct_no_change = 0
        self.majority_classifier_correction = FastBuffer(window_size)
        self.correct_no_change_correction = FastBuffer(window_size)

    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.majority_classifier = 0
        self.correct_no_change = 0
        self.confusion_matrix.restart(self.n_targets)
        self.majority_classifier_correction = FastBuffer(self.window_size)
        self.correct_no_change_correction = FastBuffer(self.window_size)

    def add_result(self, sample, prediction):
        true_y = self._get_target_index(sample, True)
        pred = self._get_target_index(prediction, True)
        old_true = self.true_labels.add_element(np.array([sample]))
        old_predict = self.predictions.add_element(np.array([prediction]))
        #print(str(old_true) + ' ' + str(old_predict))
        if (old_true is not None) and (old_predict is not None):
            self.temp += 1
            error = self.confusion_matrix.remove(self._get_target_index(old_true[0]), self._get_target_index(old_predict[0]))

            self.correct_no_change += self.correct_no_change_correction.peek()
            self.majority_classifier += self.majority_classifier_correction.peek()

            # Verify if its needed to decrease the majority_classifier count

            # Verify if its needed to decrease the correct_no_change
        if (self.get_majority_class() == sample) and (self.get_majority_class() is not None):
            self.majority_classifier += 1
            self.majority_classifier_correction.add_element([-1])
        else:
            self.majority_classifier_correction.add_element([0])

        if (self.last_true_label == sample) and (self.last_true_label is not None):
            self.correct_no_change += 1
            self.correct_no_change_correction.add_element([-1])
        else:
            self.correct_no_change_correction.add_element([0])

        self.confusion_matrix.update(true_y, pred)

        self.last_true_label = sample
        self.last_prediction = prediction

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_majority_class(self):
        """ Get the true majority class

        :return: 
        """
        if (self.n_targets is None) or (self.n_targets == 0):
            return None
        majority_class = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum = 0.0
            for j in range(self.n_targets):
                sum += self.confusion_matrix.value_at(i, j)
            sum = sum / self.true_labels.get_current_size()
            if sum > max_prob:
                max_prob = sum
                majority_class = i

        return majority_class

    def get_performance(self):
        sum = 0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            sum += self.confusion_matrix.value_at(i, i)
        return sum / self.true_labels.get_current_size()

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_performance()

    def _get_target_index(self, target, add=False):
        if (self.targets is None) and add:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        elif (self.targets is None) and (not add):
            return None
        if ((target not in self.targets) and (add)):
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        p0 = self.get_performance()
        pc = 0.0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.true_labels.get_current_size()
            sum_column = np.sum(column) / self.true_labels.get_current_size()

            pc += sum_row * sum_column

        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_t(self):
        p0 = self.get_performance()
        if self._sample_count != 0:
            pc = self.correct_no_change / self._sample_count
        else:
            pc =0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_m(self):
        p0 = self.get_performance()
        if self._sample_count != 0:
            pc = self.majority_classifier / self._sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    @property
    def _sample_count(self):
        return self.true_labels.get_current_size()

    def get_class_type(self):
        return 'collection'

    def get_info(self):
        return 'Not implemented.'

class MultiOutputMeasurements(BaseObject):
    def __init__(self, targets=None, dtype=np.int64):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = MOLConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None
        self.sample_count = 0
        self.targets = targets
        self.exact_match_count = 0
        self.j_sum = 0

    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.sample_count = 0
        self.confusion_matrix.restart(self.n_targets)
        self.exact_match_count = 0
        self.j_sum = 0
        pass

    def add_result(self, sample, prediction):
        """ Adds the result to the MOLConfusionMatrix and update exact_matches and j-index sum counts
        
        :param sample: 
        :param prediction: 
        :return: 
        """
        self.last_true_label = sample
        self.last_prediction = prediction
        m = 0
        if hasattr(sample, 'size'):
            m = sample.size
        elif hasattr(sample, 'append'):
            m = len(sample)
        self.n_targets = m
        equal = True
        for i in range(m):
            self.confusion_matrix.update(i, sample[i], prediction[i])

            # update exact_match count
            if sample[i] != prediction[i]:
                equal = False

        # update exact_match
        if equal:
            self.exact_match_count += 1

        # update j_index count
        inter = sum((sample * prediction) > 0) * 1.
        union = sum((sample + prediction) > 0) * 1.
        #print(str(inter) + ' ' + str(union))
        if union > 0:
            self.j_sum += inter / union
        elif np.sum(sample) == 0:
            self.j_sum += 1

        self.sample_count += 1

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_hamming_loss(self):
        return 1.0 - self.get_hamming_score()

    def get_hamming_score(self):
        return self.confusion_matrix.get_sum_main_diagonal() / (self.sample_count * self.n_targets)

    def get_exact_match(self):
        return self.exact_match_count / self.sample_count

    def get_j_index(self):
        return self.j_sum / self.sample_count

    def get_total_sum(self):
        return self.confusion_matrix.get_total_sum()

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    @property
    def _sample_count(self):
        return self.sample_count

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'

class WindowMultiOutputMeasurements(BaseObject):
    def __init__(self, targets=None, dtype=np.int64, window_size=200):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = MOLConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None

        self.targets = targets
        self.window_size = window_size
        self.true_labels = FastComplexBuffer(window_size, self.n_targets)
        self.predictions = FastComplexBuffer(window_size, self.n_targets)


    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix.restart(self.n_targets)
        self.exact_match_count = 0
        self.j_sum = 0
        self.true_labels = FastComplexBuffer(self.window_size, self.n_targets)
        self.predictions = FastComplexBuffer(self.window_size, self.n_targets)

    def add_result(self, sample, prediction):
        """ Adds the result to the MOLConfusionMatrix

        :param sample: 
        :param prediction: 
        :return: 
        """

        self.last_true_label = sample
        self.last_prediction = prediction
        m = 0
        if hasattr(sample, 'size'):
            m = sample.size
        elif hasattr(sample, 'append'):
            m = len(sample)
        self.n_targets = m

        for i in range(m):
            self.confusion_matrix.update(i, sample[i], prediction[i])



        old_true = self.true_labels.add_element(sample)
        old_predict = self.predictions.add_element(prediction)
        #print(old_true)
        #print(old_predict)
        # print(str(old_true) + ' ' + str(old_predict))
        if (old_true is not None) and (old_predict is not None):
            for i in range(m):
                error = self.confusion_matrix.remove(old_true[0][i], old_predict[0][i])
            # if not error:
            # print("errou")

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_hamming_loss(self):
        return 1.0 - self.get_hamming_score()

    def get_hamming_score(self):
        return metrics.hamming_score(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_exact_match(self):
        return metrics.exact_match(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_j_index(self):
        return metrics.j_index(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_total_sum(self):
        return self.confusion_matrix.get_total_sum()

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    @property
    def _sample_count(self):
        return self.true_labels.get_current_size()

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'


class RegressionMeasurements(BaseObject):
    def __init__(self):
        super().__init__()
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def add_result(self, sample, prediction):
        self.last_true_label = sample
        self.last_prediction = prediction
        self.total_square_error += (sample - prediction) * (sample - prediction)
        self.average_error += np.absolute(sample-prediction)
        self.sample_count += 1

    def get_mean_square_error(self):
        if self.sample_count == 0:
            return 0.0
        else:
            return self.total_square_error / self.sample_count

    def get_average_error(self):
        if self.sample_count == 0:
            return 0.0
        else:
            return self.average_error / self.sample_count

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_class_type(self):
        return 'collection'

    def get_info(self):
        return 'Not implemented.'


class WindowRegressionMeasurements(BaseObject):
    def __init__(self, window_size=200):
        super().__init__()
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(window_size)
        self.average_error_correction = FastBuffer(window_size)
        self.window_size = window_size

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(self.window_size)
        self.average_error_correction = FastBuffer(self.window_size)

    def add_result(self, sample, prediction):
        self.last_true_label = sample
        self.last_prediction = prediction
        self.total_square_error += (sample - prediction) * (sample - prediction)
        self.average_error += np.absolute(sample-prediction)

        old_square = self.total_square_error_correction.add_element(np.array([-1*((sample - prediction) * (sample - prediction))]))
        old_average = self.average_error_correction.add_element(np.array([-1*(np.absolute(sample-prediction))]))

        if (old_square is not None) and (old_average is not None):
            self.total_square_error += old_square[0]
            self.average_error += old_average[0]

    def get_mean_square_error(self):
        if self._sample_count == 0:
            return 0.0
        else:
            return self.total_square_error / self._sample_count

    def get_average_error(self):
        if self._sample_count == 0:
            return 0.0
        else:
            return self.average_error / self._sample_count

    def get_last(self):
        return self.last_true_label, self.last_prediction

    @property
    def _sample_count(self):
        return self.total_square_error_correction.get_current_size()

    def get_class_type(self):
        return 'collection'

    def get_info(self):
        return 'Not implemented.'