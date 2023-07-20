__all__ = [
    "Accuracy_MOA_Jaccard"
]

class Accuracy_MOA_Jaccard:
    """Multi-label metric using the Jaccard Index

    Calculates the Jaccard Index accuracy for each examples as the ratio of correct labels predicted divided by the number of predicted or expected labels.

    Note: this is one of the accuracy measures that can be found on the MOA platform. 

    Examples
    --------
    >>> from river import metrics

    >>> jaccard = metrics.multioutput.Accuracy_MOA_Jaccard()

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False}
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False}
    ... ]

    >>> for yt, yp in zip(y_true, y_pred):
    ...     jaccard.update(yt, yp)

    >>> jaccard
    0.5833333333333333

    """
    def __init__(self):
        self.sumAccuracy = 0
        self.numberEvaluations = 0

    def __repr__(self) -> str:
        return str(self.getResult())
    def __str__(self) -> str:
        return str(self.getResult())

    # Update function. Takes target and predicted label vectors and increments the accuracy sum 
    def update(self, target: dict, predicted: dict):
        # Increment the evaluation counter
        self.numberEvaluations += 1

        # Ensure we have all the necessary data in predicted dict
        # to simplify evaluation loop
        targetKeys = target.keys()
        for k in targetKeys:
            if k not in predicted.keys():
                predicted[k] = False

        # Prepare the counters
        sumReunion = 0
        sumIntersection = 0

        # Loop on predicted dictionary
        for k in predicted.keys():
            # Get values of the current label for predicted vector
            yPredicted = predicted[k]
            # Get values of the current label for target vector, defaulting to False if absent
            yTarget = target.get(k, False)
            
            # Increment the Union counter
            if(yTarget==True or yPredicted==True):
                sumReunion += 1
            # Increment the Intersection counter
            if(yTarget==True and yPredicted==True):
                sumIntersection += 1

        # Calculate the example's accuracy and increment global accuracy sum
        if(sumReunion > 0 ):
            self.sumAccuracy += float(sumIntersection)/sumReunion

    # Calculates and return the current average accuracy of evaluation for the metric
    def getResult(self):
        if self.numberEvaluations>0:
            return self.sumAccuracy/self.numberEvaluations
        else:
            return 0 