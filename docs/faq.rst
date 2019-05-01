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
