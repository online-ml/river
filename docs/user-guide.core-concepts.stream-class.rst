============
Stream class
============

.. py:currentmodule::  skmultiflow.data.base_stream

The :class:`Stream` class is in charge of "providing" data inside ``scikit-multiflow``. The most important method of the ``Stream`` class is ``next_sample(batch_size)``.

The shape :math:`(n, m)` of the :math:`X` and :math:`Y` arrays depends on the ``batch_size`` and the type of learning problem.

Supervised learning
-------------------

``next_sample(batch_size)`` will return a features vector :math:`X` and its corresponding target vector :math:`Y`

The **number of samples** :math:`n` is defined by ``batch_size`` which by default is 1.

The total number of features :math:`m` in :math:`X` is equal to the number of numerical features plus the number of categorical features: :math:`X_m = n_{num} + n_{cat}`

The number of columns :math:`m` in :math:`Y` determines the number of targets to learn. Consider the following examples:

- ``S_bc``: A binary classification stream

  - Number of targets: ``Y_m = 1``
  - Unique target values: ``[0, 1]``


- ``S_mc``: A multi-class classification stream with 3 classes (0, 1, 2)

  - Number of targets: ``Y_m = 1``
  - Unique target values: ``[0, 1, 2]``


- ``S_mc``: A multi-target classification stream, with 2 targets, where classes (0,1,2) correspond to the first target and classes (1, 2) to the second target.

  - Number of targets: ``Y_m = 2``
  - Unique target values: ``[[0, 1, 2],[1, 2]]``


- ``S_r``: A regression stream

  - Number of targets: ``Y_m = 1``
  - Target values indicates the data type: ``[float]``


- ``S_mtr``: A multi-target regression stream with 3 targets

  - Number of targets: ``Y_m = 3``
  - Target values indicates the data type: ``[float, float, float]``
