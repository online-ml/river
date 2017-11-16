Getting started
===============

In this tutorial we show how to use *scikit-multiflow*. For this example we will train a Hoeffding Tree (VFDT) with
a 'Random Tree' stream generator.

1. Create a stream
------------------

For this example we will use a the 'Random Tree' stream generator.

>>> stream = RandomTreeGenerator(n_classes=2, n_nominal_attributes=0, n_numerical_attributes=5, n_values_per_nominal=4)
>>> stream.prepare_for_use()

2. Instantiate the HoeffdingTree classifier
-------------------------------------------

We will use the default parameters.

>>> classifier = HoeffdingTree()

3. Setup a pipeline
-------------------

>>> pipe = Pipeline([('Hoeffding Tree', classifier)])

4. Setup the evaluator
----------------------

>>> eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=100000)

5. Run
------

>>> eval.eval(stream=stream, classifier=pipe)

All togheter:

>>> from skmultiflow.core.pipeline import Pipeline
>>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
>>> from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
>>> from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
>>>
>>> # 1. Create a stream
>>> stream = RandomTreeGenerator(n_classes=2, n_nominal_attributes=0, n_numerical_attributes=5, n_values_per_nominal=4)
>>> stream.prepare_for_use()
>>>
>>> # 2. Instantiate the HoeffdingTree classifier
>>> classifier = HoeffdingTree()
>>>
>>> # 3. Setup a pipeline
>>> pipe = Pipeline([('Hoeffding Tree', classifier)])
>>>
>>> # 4. Setup the evaluator
>>> eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=100000)
>>>
>>> # 5. Run
>>> eval.eval(stream=stream, classifier=pipe)