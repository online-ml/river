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
        Algorithm to track warnings and concept drifts. Attention! The drift_detector must have a warning tracker.

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
    ...     drift_detector= drift.DDM()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 86.40%

    """

    def __init__(self, model: base.Classifier, drift_detector: base.DriftDetector = None):
        self.model = model
        self.bkg_model = model.clone()
        self.drift_detector = drift_detector if drift_detector is not None else drift.DDM()

    @property
    def _wrapped_model(self):
        return self.model

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def learn_one(self, x, y):
        self._update_ddm(x, y)
        self.model.learn_one(x, y)
        return self

    def _update_ddm(self, x, y):
        y_pred = self.model.predict_one(x)
        if y_pred is None:
            return

        incorrectly_classifies = int(y_pred != y)
        self.drift_detector.update(incorrectly_classifies)

        if self.drift_detector.warning_detected:
            # If there's a warning, we train the background model
            self.bkg_model.learn_one(x, y)
        elif self.drift_detector.drift_detected:
            # If there's a drift, we replace the model with the background model
            self.model = self.bkg_model
            self.bkg_model = self.model.clone()

    @classmethod
    def _unit_test_params(cls):
        from river import linear_model, naive_bayes, preprocessing

        yield {
            "model": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
            "drift_detector": drift.DDM(),
        }
        yield {"model": naive_bayes.GaussianNB(), "drift_detector": drift.DDM()}
