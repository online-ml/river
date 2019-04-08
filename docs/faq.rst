FAQ
===

**Do all classifiers support multi-class classification?**

No, they don't. Although binary classification can be seen as a special case of multi-class classification, there are many optimizations that can be performed if we know that there are only two classes. It would be annoying to have to check whether this is the case in an online setting. All in all we find that separating both cases leads to much cleaner code.

**How may I know if a classifier supports multi-class classification?**

Each classifier that is part of ``creme`` is either a ``BinaryClassifier`` or a ``MultiClassifier``. You can use Python's ``isinstance`` to check for a particular classifier, as so:

::

    >>> from creme import base
    >>> from creme import linear_model

    >>> classifier = linear_model.LogisticRegression()
    >>> isinstance(classifier, base.BinaryClassifier)
    True
    >>> isinstance(classifier, base.MultiClassifier)
    False


**Why is ``fit_predict_one`` different from ``predict_one`` followed by ``fit_one``?**

This is a tricky question, and the answer sheds a lot of light on how online machine learning works. On the one hand ``predict_one`` makes a prediction for a set of features ``x``. On the other hand ``fit_one`` updates a model given a set of features ``x`` and a target ``y``. If you want to evaluate a model in an online manner then you need to make a prediction and then update the model so as to avoid leakage. In other words you want to call ``predict_one`` followed by ``fit_one``. If you do it the other way round then you're cheating because you're taking a look at the true ``y`` before making predicting the outcome of ``x``. All this works just fine when your model only consists of a classifier or a regressor. However, if your model is a pipeline that contains a transformer then you might want to fit the transformer before making a prediction. If you call ``predict_one`` with a pipeline then it's transformers are not fitted, which isn't necessarily desirable. However, if you call ``fit_predict_one`` then the transformers will be fitted but the final model will not.

In summary ``fit_predict_one`` takes care of calling ``fit_one`` followed by ``transform_one`` for transfomers and ``predict_one`` followed by ``fit_one`` for classifiers and regressors.
