Getting started
===============

In this tutorial we show how to use *scikit-multiflow*.

Basic example
-------------

In this first example, we will a stream to train a classifier and will evaluate it using the prequential strategy.
We will train a Hoeffding Tree (VFDT) with a 'Random Tree' stream generator.

These are the main steps:

**1. Create a stream**

For this example we will use the ``RandomTreeGenerator``. We setup the stream generator with 2 classes and 5
numerical attributes as follows:

>>> stream = RandomTreeGenerator(n_classes=2, n_numerical_attributes=5)

Before using the stream, we need to prepare it by calling ``prepare_for_use()``:

>>> stream.prepare_for_use()

**2. Instantiate the HoeffdingTree classifier**

We will use the default parameters.

>>> classifier = HoeffdingTree()

**3. Setup the evaluator**

>>> eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=100000)

* show_plot=True to get a live plot as the classifier is trained
* pretrain_size=1000 sets the minimum number of samples before the first evaluation
* max_instances=100000 to indicate the maximum number of samples to use

**4. Run the evaluation**

By calling **eval()** the *evaluator* will perform the following tasks:

* Check if there are instances in the stream
* Pass the next instance to the classifier

  - To test the classifier (using ``predict()``)
  - To update the classifier (using ``partial_fit()``)

>>> eval.eval(stream=stream, classifier=pipe)

**Putting it all together:**

::

  from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
  from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
  from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

  # 1. Create a stream
  stream = RandomTreeGenerator(n_classes=2, n_numerical_attributes=5)
  stream.prepare_for_use()

  # 2. Instantiate the HoeffdingTree classifier

  ht = HoeffdingTree()

  # 3. Setup the evaluator
  eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=100000)

  # 4. Run
  eval.eval(stream=stream, classifier=ht)