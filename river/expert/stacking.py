import typing

from river import base

__all__ = ["StackingClassifier"]


class StackingClassifier(base.EnsembleMixin, base.Classifier):
    """Stacking for binary classification.

    Parameters
    ----------
    classifiers
    meta_classifier
    include_features
        Indicates whether or not the original features should be provided to the meta-model along
        with the predictions from each model.

    Examples
    --------

    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import expert
    >>> from river import linear_model as lm
    >>> from river import metrics
    >>> from river import preprocessing as pp

    >>> dataset = datasets.Phishing()

    >>> model = compose.Pipeline(
    ...     ('scale', pp.StandardScaler()),
    ...     ('stack', expert.StackingClassifier(
    ...         classifiers=[
    ...             lm.LogisticRegression(),
    ...             lm.PAClassifier(mode=1, C=0.01),
    ...             lm.PAClassifier(mode=2, C=0.01)
    ...         ],
    ...         meta_classifier=lm.LogisticRegression()
    ...     ))
    ... )

    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 0.881387

    References
    ----------
    [^1]: [A Kaggler's Guide to Model Stacking in Practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

    """

    def __init__(
        self,
        classifiers: typing.List[base.Classifier],
        meta_classifier: base.Classifier,
        include_features=True,
    ):
        super().__init__(classifiers)
        self.meta_classifier = meta_classifier
        self.include_features = include_features

    @property
    def _multiclass(self):
        return self.meta_classifier._multiclass

    def learn_one(self, x, y):

        # Ask each model to make a prediction and then update it
        oof = {}
        for i, clf in enumerate(self):
            for k, p in clf.predict_proba_one(x).items():
                oof[f"oof_{i}_{k}"] = p
            clf.learn_one(x, y)

        # Optionally, add the base features
        if self.include_features:
            oof.update(x)

        # Update the meta-classifier using the predictions from the base classifiers
        self.meta_classifier.learn_one(oof, y)

        return self

    def predict_proba_one(self, x):

        oof = {
            f"oof_{i}_{k}": p
            for i, clf in enumerate(self)
            for k, p in clf.predict_proba_one(x).items()
        }

        if self.include_features:
            oof.update(x)

        return self.meta_classifier.predict_proba_one(oof)
