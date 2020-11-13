# Frequently Asked Questions

**Do all classifiers support multi-class classification?**

No, they don't. Although binary classification can be seen as a special case of multi-class classification, there are many optimizations that can be performed if we know that there are only two classes. It would be annoying to have to check whether this is the case in an online setting. All in all we find that separating both cases leads to much cleaner code. Note that the `multiclass` module contains wrapper models that enable you to perform multi-class classification with binary classifiers.

**How do I know if a classifier supports multi-class classification?**

Each classifier that is part of `river` is either a `base.BinaryClassifier` or a `base.MultiClassifier`. You can use Python's `isinstance` function to check for a particular classifier, as so:

```python
>>> from river import base
>>> from river import linear_model

>>> classifier = linear_model.LogisticRegression()
>>> isinstance(classifier, base.BinaryClassifier)
True
>>> isinstance(classifier, base.MultiClassifier)
False
```

**Why doesn't river do any input validation?**

Python encourages a coding style called [EAFP](https://docs.python.org/2/glossary.html?highlight=EAFP#term-eafp), which stands for "Easier to Ask for Forgiveness than Permission". The idea is to assume that runtime errors don't occur, and instead use try/expects to catch errors. The great benefit is that we don't have to drown our code with `if` statements, which is symptomatic of the [LBYL style](https://docs.python.org/2/glossary.html?highlight=EAFP#term-lbyl), which stands for "look before you leap". This makes our implementations much more readable than, say, scikit-learn, which does a lot of input validation. The catch is that users have to be careful to use sane inputs. As always, [there is no free lunch](https://www.wikiwand.com/en/No_free_lunch_theorem)!

**What about reinforcement learning?**

Reinforcement learning works in an online manner because of the nature of the task. Reinforcement learning can be therefore be seen as a subcase of online machine learning. However, we prefer not to support it because there are already many existing opensource libraries dedicated to it.

**What are the differences between scikit-learn's online learning algorithm which have a partial_fit method and their equivalents in river?**

The algorithms from `sklearn` that support incremental learning are mostly meant for mini-batch learning. In a pure streaming context where the observations arrive one by one, then `river` is much faster than `sklearn`. This is mostly because `sklearn` incurs a lot of overhead by performing data checks. Also, sklearn assumes that you're always using the same number of features. This is not the case with `river` because it use dictionaries which allows you to drop and add features as you wish.

**How do I save and load models?**

```python
>>> from river import ensemble
>>> import pickle

>>> model = ensemble.AdaptiveRandomForestClassifier()

# save
>>> with open('model.pkl', 'wb') as f:
...     pickle.dump(model, f)

# load
>>> with open('model.pkl', 'rb') as f:
...     model = pickle.load(f)
```

We also encourage you to try out [dill](https://dill.readthedocs.io/en/latest/dill.html) and [cloudpickle](https://github.com/cloudpipe/cloudpickle).

**What about neural networks?**

There are many great open-source libraries for building neural network models. We don't feel that we can bring anything of value to the existing Python ecosystem. However, we are open to implementing compatibility wrappers for popular libraries such as PyTorch and Keras.

**Who are the authors of this library?**

We are research engineers, graduate students, PhDs and machine learning researchers. The members of the develompent team are mainly located in France, Brazil and New Zealand.
