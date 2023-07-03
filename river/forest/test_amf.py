from __future__ import annotations


def test_issue_1272():
    """

    https://github.com/online-ml/river/issues/1272

    >>> import river
    >>> from river import forest, metrics

    >>> model = forest.ARFClassifier(metric=metrics.CrossEntropy())
    >>> model = model.learn_one({"x": 1}, True)

    >>> model = forest.ARFClassifier()
    >>> model = model.learn_one({"x": 1}, True)

    """
