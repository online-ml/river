=================================
Using Streams in scikit-multiflow
=================================

Stream generators
=================

.. py:currentmodule:: skmultiflow.data.agrawal_generator

Stream generators are a cheap source of data, since data samples are generated on demand we can avoid storing data physically. There are multiple stream generators in ``scikit-multiflow`` and all of them work in a similar way.

Here, we will use the :class:`AGRAWALGenerator` to exemplify how to use generators within ``scikit-multiflow``


1. Instantiate the Stream generator

   .. code-block:: python

      generator = AGRAWALGenerator()
      generator.prepare_for_use()


   The call to ``prepare_for_use()`` ensures that the Stream object is ready and **must** be done before using a Stream object.


2. Get data from the stream

   Use ``next_sample()`` to obtain data (samples) from any Stream object. The Stream will return ``n_samples`` using two arrays: ``X`` for features and ``y`` for classes (classification) or targets (regression).

   .. code-block:: python

      X, y = generator.next_sample()
      print(X.shape, y.shape)
      >>> (1, 9) (1,)


   By default, ``next_sample()`` returns one sample, but we can pass an arbitrary number of samples as ``next_sample(n_samples)``. For example, to get 1000 samples:

   .. code-block:: python

      X, y = generator.next_sample(1000)
      print(X.shape, y.shape)
      >>> (1000, 9) (1000,)

.. py:currentmodule:: skmultiflow.evaluation.evaluate_prequential


3. Check if the stream has more data

   When working with streams, it is important to know if there is more data remaining. You can use ``has_more_samples()`` to query the Stream for this information.

  .. code-block:: python

      generator.has_more_samples()
      >>> True

4. Restart the stream

   To restart a Stream object to its initial state, we can use ``restart()``

   .. code-block:: python

      generator.restart()


5: Save the data into a csv file [Optional]

   There might be cases where we want to store the information obtained from a Stream generator. An easy way to do it is using ``numpy`` and ``pandas``. First, we concatenate the ``X`` and ``y`` arrays into a single ``np.array``. Then we create a ``DataFrame`` that is easy manipulate, for example if we want to name the features, pre-process the data, etc.

  .. code-block:: python

      df = pd.DataFrame(np.hstack((X,np.array([y]).T)))


  Finally, to write the data into a csv:

  .. code-block:: python

      df.to_csv("file.csv")



**Putting it all together:**

.. code-block:: python
   :linenos:

   from skmultiflow.data import AGRAWALGenerator
   import pandas as pd
   import numpy as np

   # 1. Instantiate the stream generator
   generator = AGRAWALGenerator()
   generator.prepare_for_use()

   # 2. Get data from the stream
   X, y = generator.next_sample()
   print(X.shape, y.shape)
   >>> (1, 9) (1,)

   X, y = generator.next_sample(1000)
   print(X.shape, y.shape)
   >>> (1000, 9) (1000,)

   # 3. Check if the stream has more data
   generator.has_more_samples()
   >>> True

   # 4. Restart the stream
   generator.restart()

   # 5. Save data into a csv file [Optional]
   df = pd.DataFrame(np.hstack((X,np.array([y]).T)))