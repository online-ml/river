from . import base
from . import confusion

__all__ = ['BalancedAccuracy']

class BalancedAccuracy(base.BinaryMetric, base.MultiClassMetric):
    """Balanced Accuracy, which is the average of recall obtained on each class, 
    is used to deal with imbalanced datasets in binary and multiclass classification problems.
    
    Example:
    
        >>> from river import metrics
        >>> y_true = [True, False, True, True, False, True]
        >>> y_pred = [True, False, True, True, True, False]
        
        >>> metric = metrics.BalancedAccuracy()
        >>> for yt, yp in zip(y_true, y_pred):
        ...     metric = metric.update(yt, yp)
        
        >>> metric
        BalancedAccuracy : 62.50%
        
    """
    
    _fmt = '.2%' # will output a percentage, e.g. 0.625 will become "62,5%"
    
    @property
    def bigger_is_better(self):
        return True
    
    @property
    def requires_labels(self):
        return True
    
    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                continue
        try:
            n_classes = len(self.cm.classes)
            score = total / n_classes
            """
            if self.correction:
                chance = 1 / n_classes
                score -= chance
                score /= 1 - chance
            """
            return score
         
        except ZeroDivisionError:
            return 0.
        
        