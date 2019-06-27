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

**Why doesn't `creme` do any input validation?**

Python encourages a coding style called [EAPT]https://docs.python.org/2/glossary.html?highlight=EAFP#term-eafp, which stands for "Easier to Ask for Forgiveness than Permission". The idea is to assume that runtime errors don't occur, and instead use try/expects to catch errors. The great benefit is that we don't have to drown our code with `if` statements, which is symptomatic of the [LBYL style](https://docs.python.org/2/glossary.html?highlight=EAFP#term-lbyl), which stands for "look before you leap". This makes our implementations much more readable than, say, scikit-learn, which does a lot of input validation. The catch is that users have to be careful to use sane inputs. As always, [there is no free lunch](https://www.wikiwand.com/en/No_free_lunch_theorem)!
