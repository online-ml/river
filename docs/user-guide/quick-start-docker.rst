=============================
Quick-Start Guide with Docker
=============================

Run Hello World example
=======================

When you start scikit-multiflow container with jupyter, you will find a ``Quick Start`` jupyter notebook.

For the scikit-multiflow container with python. There are two Hello World examples.

In ``hoeffding_tree.py``, data is generated from ``WaveformGenerator``.

In ``ht_from_file.py``, data is generated from a csv file ``elec.csv``.



Start scikit-multiflow Docker container

.. code-block:: bash

    $ docker run -it skmultiflow/scikit-multiflow:latest

Run the Hoeffding Tree example :

.. code-block:: bash

    $ python hoeffding_tree.py

Write your code and run it in scikit-multiflow container
========================================================

It is possible to write and edit your code on your local machine and run it in the container.

First, create a work directory.

.. code-block:: bash

    $ mkdir workdir

Second, you get the path of that directory. This will be the value of ``hostDir``

.. code-block:: bash

    $ cd workdir && pwd

In this example, let's assume that this command returns the following path : ``/tmp/workdir``

Next, you mount the directory with the following command and start scikit-multiflow container

.. code-block:: bash

    $ docker run -it -v /tmp/workdir:/app --shm-size 2G skmultiflow/scikit-multiflow:latest

Let's download some example to ``workdir`` directory. You can also write your own code.

.. code-block:: bash

    $ wget https://raw.githubusercontent.com/scikit-multiflow/scikit-multiflow/master/docker/examples/src/hoeffding_tree.py

Now, your files are synchronized with the container. just run it.

.. code-block:: bash

    $ python hoeffding_tree.py
