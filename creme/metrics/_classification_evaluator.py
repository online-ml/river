from . import Accuracy
from . import CohenKappa, KappaM, KappaT
from . import Recall, MicroRecall, MacroRecall
from . import Precision, MicroPrecision, MacroPrecision
from . import F1, MicroF1, MacroF1
from . import GeometricMean
from . import Hamming, HammingLoss
from . import Jaccard
from . import ExactMatch
from . import Rolling
from . import ConfusionMatrix
from ._confusion_matrix import MultiLabelConfusionMatrix


class ClassificationEvaluator:
    """"Classification report

    Incrementally tracks classification performance and provide, at any moment, updated
    performance metrics. This performance evaluator is designed for single-output
    (binary and multi-class) classification tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    Examples
    --------
    >>> from creme.metrics._classification_evaluator import ClassificationEvaluator

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> evaluator = ClassificationEvaluator()

    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Classification report
    ==============================
    n_classes:		         3
    n_samples:		        25
    Accuracy:		    0.4800
    Kappa:			    0.2546
    KappaT:			   -5.5000
    KappaM:			    0.1333
    Precision:		    0.6667
    Recall:			    0.2000
    F1:				    0.3077
    GeometricMean:	    0.4463
    MicroPrecision:	    0.4800
    MicroRecall:	    0.4800
    MicroF1:		    0.4800
    MacroPrecision:	    0.5470
    MacroRecall:	    0.5111
    MacroF1:		    0.4651
    ------------------------------

    """

    # Define the format specification used for string representation.
    _fmt = '>10.4f'

    def __init__(self, cm: ConfusionMatrix = None):

        self.cm = ConfusionMatrix() if cm is None else cm
        self.accuracy = Accuracy(cm=self.cm)
        self.kappa = CohenKappa(cm=self.cm)
        self.kappa_m = KappaM(cm=self.cm)
        self.kappa_t = KappaT(cm=self.cm)
        self.recall = Recall(cm=self.cm)
        self.micro_recall = MicroRecall(cm=self.cm)
        self.macro_recall = MacroRecall(cm=self.cm)
        self.precision = Precision(cm=self.cm)
        self.micro_precision = MicroPrecision(cm=self.cm)
        self.macro_precision = MacroPrecision(cm=self.cm)
        self.f1 = F1(cm=self.cm)
        self.micro_f1 = MicroF1(cm=self.cm)
        self.macro_f1 = MacroF1(cm=self.cm)
        self.geometric_mean = GeometricMean(cm=self.cm)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def accuracy_score(self):
        return self.accuracy.get()

    def kappa_score(self):
        return self.kappa.get()

    def kappa_t_score(self):
        return self.kappa_t.get()

    def kappa_m_score(self):
        return self.kappa_m.get()

    def precision_score(self):
        return self.precision.get()

    def recall_score(self):
        return self.recall.get()

    def f1_score(self):
        return self.f1.get()

    def geometric_mean_score(self):
        return self.geometric_mean.get()

    def get_last(self):
        return self.cm.last_y_true, self.cm.last_y_pred

    @property
    def n_samples(self):
        return self.cm.n_samples

    @property
    def n_classes(self):
        return self.cm.n_classes

    @property
    def confusion_matrix(self):
        return self.cm

    def reset(self):
        self.cm.reset()

    def __repr__(self):
        t_line = '=============================='
        b_line = '------------------------------'
        return ''.join([
            'Classification report\n',
            f'{t_line}\n',
            f'n_classes:\t\t{self.n_classes:>10}\n',
            f'n_samples:\t\t{self.n_samples:>10}\n',
            '\n',
            self._info(),
            f'{b_line}',
        ])

    def _info(self):
        return ''.join([
            f'Accuracy:\t\t{self.accuracy.get():{self._fmt}}\n',
            f'Kappa:\t\t\t{self.kappa.get():{self._fmt}}\n',
            f'KappaT:\t\t\t{self.kappa_t.get():{self._fmt}}\n',
            f'KappaM:\t\t\t{self.kappa_m.get():{self._fmt}}\n',
            f'Precision:\t\t{self.precision.get():{self._fmt}}\n',
            f'Recall:\t\t\t{self.recall.get():{self._fmt}}\n',
            f'F1:\t\t\t\t{self.f1.get():{self._fmt}}\n',
            f'GeometricMean:\t{self.geometric_mean.get():{self._fmt}}\n',
            '\n',
            f'MicroPrecision:\t{self.micro_precision.get():{self._fmt}}\n',
            f'MicroRecall:\t{self.micro_recall.get():{self._fmt}}\n',
            f'MicroF1:\t\t{self.micro_f1.get():{self._fmt}}\n',
            f'MacroPrecision:\t{self.macro_precision.get():{self._fmt}}\n',
            f'MacroRecall:\t{self.macro_recall.get():{self._fmt}}\n',
            f'MacroF1:\t\t{self.macro_f1.get():{self._fmt}}\n',
        ])


class WindowClassificationEvaluator(ClassificationEvaluator):
    """Rolling classification report

    Incrementally tracks classification performance over a sliding window and provide,
    at any moment, updated performance metrics. This performance evaluator is designed
    for single-output (binary and multi-class) classification tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    window_size: int
        Window size.

    Examples
    --------
    >>> from creme.metrics._classification_evaluator import WindowClassificationEvaluator

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> evaluator = WindowClassificationEvaluator(window_size=20)

    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Classification report [rolling]
    ==============================
    n_classes:		         3
    n_samples:		        20
    window_size:	        20
    Accuracy:		    0.4000
    Kappa:			    0.1696
    KappaT:			   -5.0000
    KappaM:			    0.2000
    Precision:		    0.6667
    Recall:			    0.2000
    F1:				    0.3077
    GeometricMean:	    0.0000
    MicroPrecision:	    0.4000
    MicroRecall:	    0.4000
    MicroF1:		    0.4000
    MacroPrecision:	    0.4722
    MacroRecall:	    0.2889
    MacroF1:		    0.3379
    ------------------------------

    """

    def __init__(self, cm: ConfusionMatrix = None, window_size=200):
        self.window_size = window_size
        self._rolling_cm = Rolling(ConfusionMatrix() if cm is None else cm,
                                   window_size=self.window_size)
        super().__init__(cm=self._rolling_cm.metric)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self._rolling_cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def __repr__(self):
        t_line = '=============================='
        b_line = '------------------------------'
        return ''.join([
            'Classification report [rolling]\n',
            f'{t_line}\n',
            f'n_classes:\t\t{self.n_classes:>10}\n',
            f'n_samples:\t\t{self.n_samples:>10}\n',
            f'window_size:\t{self.window_size:>10}\n',
            '\n',
            self._info(),
            f'{b_line}',
            ])


class MLClassificationEvaluator:
    """Multi-label classification report.

    Incrementally tracks a classifier's performance and provide, at any moment, updated
    performance metrics. This performance evaluator is designed for multi-output
    (multi-label) classification tasks.

    Parameters
    ----------
    cm: MultiLabelConfusionMatrix, optional (default=None)
        Multi-label confusion matrix instance.

    Examples
    --------
    >>> from creme.metrics._classification_evaluator import MLClassificationEvaluator

    >>> y_0 = [True]*100
    >>> y_1 = [True]*90 + [False]*10
    >>> y_2 = [True]*85 + [False]*10 + [True]*5
    >>> y_true = []
    >>> y_pred = []
    >>> for i in range(len(y_0)):
    ...     y_true.append({0:True, 1:True, 2:True})
    ...     y_pred.append({0:y_0[i], 1:y_1[i], 2:y_2[i]})

    >>> evaluator = MLClassificationEvaluator()

    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Multi-label classification report
    ==============================
    n_classes:		         3
    n_samples:		       100
    Hamming:		    0.9333
    HammingLoss:	    0.0667
    ExactMatch:		    0.8500
    JaccardIndex:	    0.9333
    ------------------------------

    """

    # Define the format specification used for string representation.
    _fmt = '>10.4f'

    def __init__(self, cm: MultiLabelConfusionMatrix = None):
        self.cm = MultiLabelConfusionMatrix() if cm is None else cm
        self.hamming = Hamming(cm=self.cm)
        self.hamming_loss = HammingLoss(cm=self.cm)
        self.jaccard_index = Jaccard(cm=self.cm)
        self.exact_match = ExactMatch(cm=self.cm)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def hamming_score(self):
        return self.hamming.get()

    def hamming_loss_score(self):
        return self.hamming_loss.get()

    def exact_match_score(self):
        return self.exact_match.get()

    def jaccard_score(self):
        return self.jaccard_index.get()

    def get_last(self):
        return self.cm.last_y_true, self.cm.last_y_pred

    @property
    def n_samples(self):
        return self.cm.n_samples

    @property
    def n_labels(self):
        return self.cm.n_labels

    @property
    def confusion_matrix(self):
        return self.cm

    def reset(self):
        self.cm.reset()

    def __repr__(self):
        t_line = '=============================='
        b_line = '------------------------------'
        return ''.join([
            'Multi-label classification report\n',
            f'{t_line}\n',
            f'n_classes:\t\t{self.n_labels:>10}\n',
            f'n_samples:\t\t{self.n_samples:>10}\n',
            '\n',
            self._info(),
            f'{b_line}',
        ])

    def _info(self):
        return ''.join([
            f'Hamming:\t\t{self.hamming.get():{self._fmt}}\n',
            f'HammingLoss:\t{self.hamming_loss.get():{self._fmt}}\n',
            f'ExactMatch:\t\t{self.exact_match.get():{self._fmt}}\n',
            f'JaccardIndex:\t{self.jaccard_index.get():{self._fmt}}\n',
        ])


class WindowMLClassificationEvaluator(MLClassificationEvaluator):
    """Multi-label classification report.

    Incrementally tracks a classifier's performance over a sliding window and provide,
    at any moment, updated performance metrics. This performance evaluator is designed
    for multi-output (multi-label) classification tasks.

    Parameters
    ----------
    cm: MultiLabelConfusionMatrix, optional (default=None)
        Multi-label confusion matrix instance.

    window_size: int
        Window size.

    Examples
    --------
    >>> from creme.metrics._classification_evaluator import WindowMLClassificationEvaluator

    >>> y_0 = [True]*100
    >>> y_1 = [True]*90 + [False]*10
    >>> y_2 = [True]*85 + [False]*10 + [True]*5
    >>> y_true = []
    >>> y_pred = []
    >>> for i in range(len(y_0)):
    ...     y_true.append({0:True, 1:True, 2:True})
    ...     y_pred.append({0:y_0[i], 1:y_1[i], 2:y_2[i]})

    >>> evaluator = WindowMLClassificationEvaluator(window_size=20)
    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Multi-label classification report [rolling]
    ==============================
    n_labels:		         3
    n_samples:		        20
    window_size:	        20
    Hamming:		    0.6667
    HammingLoss:	    0.3333
    ExactMatch:		    0.2500
    JaccardIndex:	    0.6667
    ------------------------------

    """

    def __init__(self, cm: ConfusionMatrix = None, window_size=200):
        self.window_size = window_size
        self._rolling_cm = Rolling(MultiLabelConfusionMatrix() if cm is None else cm,
                                   window_size=self.window_size)
        super().__init__(cm=self._rolling_cm.metric)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self._rolling_cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def __repr__(self):
        t_line = '=============================='
        b_line = '------------------------------'
        return ''.join([
            'Multi-label classification report [rolling]\n',
            f'{t_line}\n',
            f'n_labels:\t\t{self.n_labels:>10}\n',
            f'n_samples:\t\t{self.n_samples:>10}\n',
            f'window_size:\t{self.window_size:>10}\n',
            '\n',
            self._info(),
            f'{b_line}',
            ])
