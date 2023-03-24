from river import base

from .base import ActiveLearningClassifier


class FixedUncertainty(ActiveLearningClassifier):

    """Strategy of Active Learning to select instances more significative based on uncertainty.

    The fixed uncertainty sampler selects samples for labeling based on the uncertainty of the prediction.
    The higher the uncertainty, the more likely the sample will be selected for labeling. The uncertainty
    measure is compared with a fixed uncertainty limit.

    The FixedUncertainty use the maximium posterior probability.
    So use only the predict_proba_one(X).
    Do not use predict_one(x).

    Version 1.0.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    theta
        Threshold of uncertainty. If posteriori is less to theta. So, the instance is selected for labeling.
        Default value: 0.95.
        More information in the paper of reference.
    seed
        Random number generator seed for reproducibility.

    >>> from river import active
    >>> from river import datasets
    >>> from river import feature_extraction
    >>> from river import linear_model

    >>> dataset = datasets.SMSSpam()

    >>> model = (
    ...       feature_extraction.TFIDF(on='body') |
    ...        linear_model.LogisticRegression()
    ...     )
    >>> model = active.FixedUncertainty(model, seed=42)

    >>> for x, y in dataset:

            ## IF ask = True
            ##    Selected instance.
            ## IF ask = False
            ##    Instance not selected
            ## The FixedUncertainty use the maximium
            ## posterior probability. So use only the predict_proba_one(X).
            ## Do not use predict_one(x).
            >>> y_pred, ask = model.predict_proba_one(x)


    References
    ----------
    [^1]: I. Zliobaite, A. Bifet, B.Pfahringer, G. Holmes. “Active Learning with Drifting Streaming Data”, IEEE Transactions on Neural Netowrks and Learning Systems, Vol.25 (1), pp.27-39, 2014.


    """

    def __init__(self, classifier: base.Classifier, theta: float = 0.95, seed=None):
        super().__init__(classifier, seed=seed)
        self.theta = theta

    def _ask_for_label(self, x, y_pred) -> bool:
        """Ask for the label of a current instance.

        Based on the uncertainty of the base classifier, it checks whether the current instance should be labeled.

        Parameters
        ----------
        x
            Instance

        y_pred

            Arrays of predicted labels


        Returns
        -------
        selected
            A boolean indicating whether a label is needed.
            True for selected instance.
            False for not selecte instance.

        """
        maximum_posteriori = max(y_pred.values())
        selected = False
        if maximum_posteriori < self.theta:
            selected = True
        return selected
