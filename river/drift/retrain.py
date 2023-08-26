from __future__ import annotations

from river import base, drift


class DriftRetrainingClassifier(base.Wrapper, base.Classifier):
    """Drift retraining classifier.

    This classifier is a wrapper for any classifier. It monitors the incoming data for concept
    drifts and warnings in the model's accurary. In case a warning is detected, a background model
    starts to train. If a drift is detected, the model will be replaced by the background model,
    and the background model will be reset.

    Parameters
    ----------
    model
        The classifier and background classifier class.
    drift_detector
        Algorithm to track warnings and concept drifts. Attention! If the parameter train_in_background is True, the drift_detector must have a warning tracker.
    train_in_background
        Parameter to determine if a background model will be used.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import drift
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Elec2().take(3000)

    >>> model = drift.DriftRetrainingClassifier(
    ...     model=tree.HoeffdingTreeClassifier(),
    ...     drift_detector=drift.binary.DDM()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 86.46%

    """

    def __init__(
        self,
        model: base.Classifier,
        drift_detector: base.DriftAndWarningDetector
        | base.BinaryDriftAndWarningDetector
        | None = None,
        train_in_background: bool = True,
    ):
        self.model = model
        self.train_in_background = train_in_background
        self.drift_detector = drift_detector if drift_detector is not None else drift.binary.DDM()
        if self.train_in_background:
            self.bkg_model = model.clone()

    @property
    def _wrapped_model(self):
        return self.model

    def predict_proba_one(self, x, **kwargs):
        return self.model.predict_proba_one(x, **kwargs)

    def learn_one(self, x, y, **kwargs):
        self._update_detector(x, y)
        self.model.learn_one(x, y, **kwargs)
        return self

    def _update_detector(self, x, y):
        y_pred = self.model.predict_one(x)
        if y_pred is None:
            return

        incorrectly_classifies = int(y_pred != y)
        self.drift_detector.update(incorrectly_classifies)

        if self.train_in_background:
            if self.drift_detector.warning_detected:
                # If there's a warning, we train the background model
                self.bkg_model.learn_one(x, y)
            elif self.drift_detector.drift_detected:
                # If there's a drift, we replace the model with the background model
                self.model = self.bkg_model
                self.bkg_model = self.model.clone()
        else:
            if self.drift_detector.drift_detected:
                # If there's a drift, we reset the model
                self.model = self.model.clone()

    @classmethod
    def _unit_test_params(cls):
        from river import linear_model, naive_bayes, preprocessing

        yield {
            "model": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
            "drift_detector": drift.binary.DDM(),
        }
        yield {"model": naive_bayes.GaussianNB(), "drift_detector": drift.binary.DDM()}
